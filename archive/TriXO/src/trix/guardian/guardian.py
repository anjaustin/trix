"""
Training Observer - Adaptive Training with Self-Observation

The Training Observer monitors training dynamics and applies targeted
interventions when the model struggles. It combines:

- Observation: Track routing patterns, gradient flow, and loss trajectories
- Prediction: Anticipate which operations will struggle
- Intervention: Apply minimal corrections to tile signatures

Key design principle: Errors are information, not failures. The observer
treats suboptimal trajectories as signals indicating where adjustment helps.

This is experimental research code. The core trix.nn layers work without it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .observer import ObserverModel, ObservationFrame, ObservationBuffer, StateEncoder
from .programmable_tile import ProgrammableTileBank
from .reflector import SuperpositionedReflector, XORReflector, TrainingManifoldReflector


@dataclass
class InterventionRecord:
    """Record of an observer intervention."""
    epoch: int
    step: int
    level: int
    confidence: float
    tiles_modified: int
    max_blend: float
    reason: str
    message: str
    before_accuracy: float
    after_accuracy: Optional[float] = None


class TrainingObserver(nn.Module):
    """
    Training Observer - Observation + Reflection + Intervention

    Combines multiple components for adaptive training:

    1. Observe: Track training dynamics (routing, gradients, loss)
    2. Reflect: Analyze what changed and whether it helped (XOR reflection)
    3. Predict: Anticipate where the model will struggle
    4. Intervene: Apply minimal corrections to tile signatures
    5. Detect success: Recognize when training is progressing well

    The observer uses bounded intervention (max_blend parameter) to ensure
    corrections are gentle and don't destabilize training.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_tiles: int = 16,
        num_ops: int = 8,
        hidden_dim: int = 128,
        state_dim: int = 64,
        window_size: int = 20,
        num_reflection_bases: int = 4,
        intervention_threshold: float = 0.7,
        max_blend: float = 0.1,
        gentleness: float = None,  # Backwards compatibility alias for max_blend
    ):
        super().__init__()

        # Handle backwards compatibility: gentleness overrides max_blend if provided
        if gentleness is not None:
            max_blend = gentleness

        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_ops = num_ops
        self.window_size = window_size
        self.intervention_threshold = intervention_threshold
        self.max_blend = max_blend
        # Backwards compatibility alias
        self.gentleness = max_blend

        # Core Components
        
        # State encoder
        self.state_encoder = StateEncoder(
            obs_dim=11 + num_tiles + num_ops,  # scalars + tiles + ops
            hidden_dim=hidden_dim,
            state_dim=state_dim
        )
        
        # Observer model
        self.observer = ObserverModel(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_tiles=num_tiles,
            num_ops=num_ops,
            d_model=d_model
        )
        
        # Superpositioned Reflector (multi-angle self-view)
        self.reflector = SuperpositionedReflector(
            d_model=d_model,
            num_bases=num_reflection_bases,
            learnable_combination=True
        )
        
        # XOR Reflector (retrospective analysis)
        self.xor_reflector = XORReflector(d_model)
        
        # Training manifold reflector (meta-level)
        self.trajectory_reflector = TrainingManifoldReflector(
            state_dim=state_dim,
            hidden_dim=hidden_dim
        )

        # Tracking
        self.observation_buffer = ObservationBuffer(
            max_size=1000,
            window_size=window_size
        )
        
        self.intervention_history: List[InterventionRecord] = []
        self.success_count = 0
        self.total_interventions = 0

        # Previous state for XOR comparison
        self._prev_state: Optional[torch.Tensor] = None

    # Observation

    def observe(self, frame: ObservationFrame):
        """Add observation to buffer."""
        self.observation_buffer.add(frame)
    
    def get_observation_window(self) -> torch.Tensor:
        """Get recent observations as encoded tensor."""
        return self.observation_buffer.get_recent_window(encoder=self.state_encoder)

    # Reflection

    def reflect(self, current_repr: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Apply superpositioned reflection to representation.

        Projects the representation through multiple learned bases to provide
        different "views" of the current state.
        """
        return self.reflector(current_repr)

    def reflect_on_change(
        self,
        current_repr: torch.Tensor,
        previous_repr: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        XOR reflection: assess what changed between states.

        Returns analysis of the delta including magnitude and direction.
        """
        if previous_repr is None:
            previous_repr = self._prev_state
        
        if previous_repr is None:
            # First observation, no change to assess
            self._prev_state = current_repr.detach().clone()
            return {'message': 'First observation', 'magnitude': 0.0}
        
        # Handle shape mismatches (different batch sizes)
        if current_repr.shape != previous_repr.shape:
            # Use mean representation for comparison
            current_mean = current_repr.mean(dim=0, keepdim=True)
            previous_mean = previous_repr.mean(dim=0, keepdim=True)
            _, info = self.xor_reflector(current_mean, previous_mean)
        else:
            _, info = self.xor_reflector(current_repr, previous_repr)
        
        # Update previous state
        self._prev_state = current_repr.detach().clone()
        
        return info

    def assess_trajectory(self) -> Dict:
        """
        Assess overall training trajectory.

        Uses the trajectory reflector to evaluate recent training history.
        """
        if len(self.observation_buffer.frames) < self.window_size:
            return {'message': 'Gathering observations...', 'good_prob': 0.5}
        
        state_history = self.get_observation_window()
        _, info = self.trajectory_reflector(state_history)
        
        return info

    # Prediction & Decision

    def predict(self) -> Dict:
        """
        Predict where intervention might help.

        Returns predictions about error risk and recommended intervention level.
        """
        if len(self.observation_buffer.frames) < 5:
            return {
                'ready': False,
                'message': 'Need more observations',
            }
        
        state_history = self.get_observation_window()
        decision = self.observer.get_intervention_decision(
            state_history,
            confidence_threshold=self.intervention_threshold
        )
        decision['ready'] = True
        
        return decision
    
    def should_intervene(self) -> Tuple[bool, Dict]:
        """
        Decide whether intervention is warranted.

        Only intervenes when confidence is high and training isn't going well.
        """
        prediction = self.predict()

        if not prediction.get('ready', False):
            return False, prediction

        # Don't intervene if training is succeeding
        if prediction.get('celebrating', False):
            self.success_count += 1
            return False, prediction

        return prediction.get('intervene', False), prediction

    # Intervention

    def intervene(
        self,
        tile_bank: ProgrammableTileBank,
        prediction: Dict,
        epoch: int = 0,
        step: int = 0,
        current_accuracy: float = 0.0
    ) -> InterventionRecord:
        """
        Apply intervention to tile bank.

        Blends correction signals into tile signatures, bounded by max_blend.
        """
        level = prediction.get('level', 0)
        
        if level == 0:
            # No intervention
            record = InterventionRecord(
                epoch=epoch, step=step, level=0,
                confidence=prediction.get('confidence', 0),
                tiles_modified=0, max_blend=0,
                reason='no_intervention',
                message=prediction.get('message', ''),
                before_accuracy=current_accuracy
            )
            return record
        
        # Get correction parameters
        corrections = prediction.get('tile_corrections')
        blend_factors = prediction.get('blend_factors')
        
        if corrections is None or blend_factors is None:
            record = InterventionRecord(
                epoch=epoch, step=step, level=level,
                confidence=prediction.get('confidence', 0),
                tiles_modified=0, max_blend=0,
                reason='missing_corrections',
                message='Prediction missing correction data',
                before_accuracy=current_accuracy
            )
            return record
        
        # Apply blend cap
        blend_factors = blend_factors.clamp(max=self.max_blend)
        
        # Level-specific intervention
        if level >= 4:  # Signature surgery
            tiles_modified = tile_bank.apply_signature_corrections(
                corrections=corrections,
                blends=blend_factors,
                reason=f'guardian_L{level}_e{epoch}_s{step}'
            )
        else:
            # Lower levels (1-3) would affect training differently
            # For now, we implement signature-level intervention
            tiles_modified = tile_bank.apply_signature_corrections(
                corrections=corrections * (level / 5),  # Scale by level
                blends=blend_factors * (level / 5),
                reason=f'guardian_L{level}_e{epoch}_s{step}'
            )
        
        self.total_interventions += 1
        
        record = InterventionRecord(
            epoch=epoch, step=step, level=level,
            confidence=prediction.get('confidence', 0),
            tiles_modified=tiles_modified,
            max_blend=blend_factors.max().item(),
            reason=f'level_{level}_intervention',
            message=prediction.get('message', ''),
            before_accuracy=current_accuracy
        )
        
        self.intervention_history.append(record)

        return record

    # Main Interface

    def step(
        self,
        tile_bank: ProgrammableTileBank,
        observation: ObservationFrame,
        current_repr: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Main step: observe, reflect, decide, and optionally intervene.

        Call this each training step. Returns summary of observations and actions.
        """
        # Observe
        self.observe(observation)
        
        # Reflect on change (if we have representation)
        change_info = {}
        if current_repr is not None:
            change_info = self.reflect_on_change(current_repr)
        
        # Assess trajectory
        trajectory_info = self.assess_trajectory()
        
        # Decide
        should_act, prediction = self.should_intervene()
        
        result = {
            'observed': True,
            'change_info': change_info,
            'trajectory_info': trajectory_info,
            'prediction': prediction,
            'intervened': False,
            'intervention_record': None,
            'message': prediction.get('message', ''),
            'celebrating': prediction.get('celebrating', False),
        }
        
        # Intervene if warranted
        if should_act:
            record = self.intervene(
                tile_bank=tile_bank,
                prediction=prediction,
                epoch=observation.epoch,
                step=observation.step,
                current_accuracy=observation.accuracy
            )
            result['intervened'] = True
            result['intervention_record'] = record
        
        return result
    
    # Statistics

    def get_stats(self) -> Dict:
        """Get observer statistics."""
        trajectory_stats = self.observation_buffer.get_trajectory_stats()

        interventions_by_level = {}
        for record in self.intervention_history:
            level = record.level
            interventions_by_level[level] = interventions_by_level.get(level, 0) + 1

        return {
            'total_observations': len(self.observation_buffer.frames),
            'total_interventions': self.total_interventions,
            'success_count': self.success_count,
            'celebration_count': self.success_count,  # Backwards compatibility
            'interventions_by_level': interventions_by_level,
            'trajectory': trajectory_stats,
            'reflector_diversity': self.reflector.analyze_diversity(),
            'max_blend': self.max_blend,
            'gentleness_setting': self.max_blend,  # Backwards compatibility
            'intervention_threshold': self.intervention_threshold,
        }

    @property
    def celebration_count(self) -> int:
        """Backwards compatibility alias for success_count."""
        return self.success_count

    def get_celebration_rate(self) -> float:
        """Fraction of observations that detected successful training."""
        total = len(self.observation_buffer.frames)
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def get_intervention_rate(self) -> float:
        """What fraction of observations resulted in intervention?"""
        total = len(self.observation_buffer.frames)
        if total == 0:
            return 0.0
        return self.total_interventions / total
    
    # Persistence

    def get_state_for_save(self) -> Dict:
        """Get state dict for saving."""
        return {
            'observer': self.observer.state_dict(),
            'reflector': self.reflector.state_dict(),
            'xor_reflector': self.xor_reflector.state_dict(),
            'trajectory_reflector': self.trajectory_reflector.state_dict(),
            'state_encoder': self.state_encoder.state_dict(),
            'intervention_history': [
                vars(r) for r in self.intervention_history[-100:]  # Keep last 100
            ],
            'stats': {
                'total_interventions': self.total_interventions,
                'success_count': self.success_count,
            }
        }

    def load_state_from_save(self, state: Dict):
        """Load state from saved dict."""
        self.observer.load_state_dict(state['observer'])
        self.reflector.load_state_dict(state['reflector'])
        self.xor_reflector.load_state_dict(state['xor_reflector'])
        self.trajectory_reflector.load_state_dict(state['trajectory_reflector'])
        self.state_encoder.load_state_dict(state['state_encoder'])

        if 'stats' in state:
            self.total_interventions = state['stats'].get('total_interventions', 0)
            # Handle both old and new key names
            self.success_count = state['stats'].get('success_count',
                state['stats'].get('celebration_count', 0))


# Backwards compatibility alias
GuardianAngel = TrainingObserver
