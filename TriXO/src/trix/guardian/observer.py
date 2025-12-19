"""
Observer Model - Watching Learning Dynamics

The Observer sees everything:
- Routing decisions
- Signature movements
- Gradient flows
- Performance trajectories

It builds a model of how learning works, not just what is learned.
Then it predicts where entropy is flowing - where the model needs gentle guidance.

"Wrong is just a signal. Distributed entropy signaling the correct direction."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class ObservationFrame:
    """
    A snapshot of training dynamics at one moment.
    
    This is what the Observer sees - full transparency into the learning process.
    """
    epoch: int
    step: int
    
    # === Routing Dynamics ===
    routing_scores: Optional[torch.Tensor] = None  # [batch, num_tiles]
    routing_entropy: float = 0.0
    tile_activations: Optional[torch.Tensor] = None  # [num_tiles] counts
    
    # === Signature State ===
    signature_positions: Optional[torch.Tensor] = None  # [num_tiles, d_model]
    signature_movement: float = 0.0  # Total movement from init
    signature_velocity: Optional[torch.Tensor] = None  # [num_tiles, d_model] delta
    
    # === Gradients ===
    gradient_norm: float = 0.0
    gradient_norms_per_layer: Dict[str, float] = field(default_factory=dict)
    
    # === Performance ===
    loss: float = 0.0
    accuracy: float = 0.0
    per_op_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # === Manifold ===
    curvature: float = 0.0
    tile_purity: float = 0.0
    
    # === Reflector State ===
    xor_delta_magnitude: float = 0.0
    superposition_diversity: float = 0.0
    
    def to_tensor(self, d_model: int = 128, num_tiles: int = 16) -> torch.Tensor:
        """Convert observation to fixed-size tensor for neural processing."""
        features = []
        
        # Scalars
        features.extend([
            self.epoch / 100,  # Normalize
            self.step / 1000,
            self.routing_entropy,
            self.signature_movement,
            self.gradient_norm,
            self.loss,
            self.accuracy / 100,
            self.curvature,
            self.tile_purity,
            self.xor_delta_magnitude,
            self.superposition_diversity,
        ])
        
        # Per-tile features (flattened)
        if self.tile_activations is not None:
            act = self.tile_activations.float()
            act = act / (act.sum() + 1e-8)  # Normalize to distribution
            features.extend(act.tolist())
        else:
            features.extend([1.0 / num_tiles] * num_tiles)
        
        # Per-op accuracy (8 ops for 6502)
        ops = ['ADC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'INC', 'DEC']
        for op in ops:
            features.append(self.per_op_accuracy.get(op, 0.5) / 100)
        
        return torch.tensor(features, dtype=torch.float32)


class StateEncoder(nn.Module):
    """
    Encode observation frames into fixed-size state vectors.
    
    The encoder compresses the rich observation into a form
    the Observer model can process temporally.
    """
    
    def __init__(
        self, 
        obs_dim: int = 35,  # 11 scalars + 16 tiles + 8 ops
        hidden_dim: int = 128,
        state_dim: int = 64
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Encode observation tensor to state vector."""
        return self.encoder(obs_tensor)
    
    def encode_frame(self, frame: ObservationFrame) -> torch.Tensor:
        """Convenience: encode ObservationFrame directly."""
        obs_tensor = frame.to_tensor()
        return self.forward(obs_tensor)


class ObserverModel(nn.Module):
    """
    The Observer - watches learning dynamics and predicts where guidance is needed.
    
    Architecture:
    1. Temporal encoder (LSTM) processes history of observations
    2. Error predictor estimates P(error) per operation
    3. Intervention decider chooses intervention level (0-5)
    4. Tile programmer generates correction parameters
    
    The Observer is trained to:
    - Predict which operations will struggle
    - Know when to intervene (and when to let the model learn)
    - Generate gentle corrections when needed
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        hidden_dim: int = 128,
        num_tiles: int = 16,
        num_ops: int = 8,
        d_model: int = 128
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_tiles = num_tiles
        self.num_ops = num_ops
        self.d_model = d_model
        
        # State encoder
        self.state_encoder = StateEncoder(state_dim=state_dim)
        
        # Temporal encoder - understand trajectory, not just current state
        self.temporal = nn.LSTM(
            state_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # === Prediction Heads ===
        
        # Error predictor: which operations are at risk?
        self.error_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_ops),
        )
        
        # Intervention decider: what level of intervention?
        # 0: None, 1: Nudge repr, 2: Mod gradient, 3: Adjust lr, 4: Sig surgery, 5: Weight surgery
        self.intervention_decider = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 6),
        )
        
        # Tile programmer: which tiles need correction?
        self.tile_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tiles),
        )
        
        # Correction generator: what direction to nudge?
        self.correction_generator = nn.Sequential(
            nn.Linear(hidden_dim + num_tiles, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tiles * d_model),
        )
        
        # Blend factor generator: how much to nudge?
        self.blend_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_tiles),
            nn.Sigmoid(),  # Blend in [0, 1]
        )
        
        # Celebration detector: is the model doing well?
        self.celebration_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self, 
        state_history: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process observation history and generate predictions/interventions.
        
        state_history: [seq_len, state_dim] or [batch, seq_len, state_dim]
        
        Returns dict with:
            - error_probs: [num_ops] probability of error per operation
            - intervention_logits: [6] logits for intervention level
            - tile_corrections: [num_tiles, d_model] correction directions
            - blend_factors: [num_tiles] how much to apply corrections
            - celebration_score: [1] how well is training going?
        """
        # Handle unbatched input
        if state_history.dim() == 2:
            state_history = state_history.unsqueeze(0)  # [1, seq, state_dim]
        
        # Temporal encoding
        _, (h, _) = self.temporal(state_history)
        context = h[-1]  # [batch, hidden_dim]
        
        # Predictions
        error_probs = torch.sigmoid(self.error_predictor(context))
        intervention_logits = self.intervention_decider(context)
        
        # Tile selection and correction
        tile_logits = self.tile_selector(context)
        tile_weights = torch.softmax(tile_logits, dim=-1)
        
        # Generate corrections conditioned on tile selection
        correction_input = torch.cat([context, tile_weights], dim=-1)
        corrections_flat = self.correction_generator(correction_input)
        tile_corrections = corrections_flat.view(-1, self.num_tiles, self.d_model)
        
        # Blend factors (gentleness parameter)
        blend_factors = self.blend_generator(context) * 0.2  # Cap at 0.2 for gentleness
        
        # Celebration score
        celebration = self.celebration_detector(context)
        
        result = {
            'error_probs': error_probs.squeeze(0),
            'intervention_logits': intervention_logits.squeeze(0),
            'tile_corrections': tile_corrections.squeeze(0),
            'blend_factors': blend_factors.squeeze(0),
            'celebration_score': celebration.squeeze(),
            'tile_weights': tile_weights.squeeze(0),
        }
        
        if return_all:
            result['context'] = context.squeeze(0)
        
        return result
    
    def get_intervention_decision(
        self,
        state_history: torch.Tensor,
        confidence_threshold: float = 0.7
    ) -> Dict:
        """
        Make intervention decision with confidence check.
        
        Returns decision dict including whether to intervene.
        """
        predictions = self.forward(state_history)
        
        # Get intervention level
        intervention_probs = F.softmax(predictions['intervention_logits'], dim=-1)
        level = intervention_probs.argmax().item()
        confidence = intervention_probs[level].item()
        
        # Check error predictions
        max_error_prob = predictions['error_probs'].max().item()
        at_risk_ops = (predictions['error_probs'] > 0.5).sum().item()
        
        # Celebration check
        celebrating = predictions['celebration_score'].item() > 0.7
        
        decision = {
            'intervene': confidence > confidence_threshold and level > 0 and not celebrating,
            'level': level,
            'confidence': confidence,
            'max_error_prob': max_error_prob,
            'at_risk_ops': at_risk_ops,
            'celebrating': celebrating,
            'tile_corrections': predictions['tile_corrections'],
            'blend_factors': predictions['blend_factors'],
            'message': self._get_message(level, celebrating, confidence),
        }
        
        return decision
    
    def _get_message(self, level: int, celebrating: bool, confidence: float) -> str:
        """Generate Guardian Angel message."""
        if celebrating:
            return "🔥 Yeah, look at 'em go!"
        elif level == 0:
            return "👀 Watching... all good"
        elif level == 1:
            return "🤲 Gentle nudge incoming"
        elif level == 2:
            return "📐 Adjusting gradient flow"
        elif level == 3:
            return "⚡ Tuning learning rate"
        elif level == 4:
            return "🔧 Signature surgery"
        elif level == 5:
            return "⚙️ Weight surgery (careful now)"
        else:
            return f"Level {level}, confidence {confidence:.2f}"


class ObservationBuffer:
    """
    Buffer to collect and manage observation history.
    
    Provides windowed access for the Observer model.
    """
    
    def __init__(self, max_size: int = 1000, window_size: int = 20):
        self.max_size = max_size
        self.window_size = window_size
        self.frames: List[ObservationFrame] = []
        self.encoded_cache: Dict[int, torch.Tensor] = {}
        
    def add(self, frame: ObservationFrame):
        """Add new observation frame."""
        self.frames.append(frame)
        if len(self.frames) > self.max_size:
            # Remove oldest
            self.frames.pop(0)
            # Clear old cache entries
            self.encoded_cache = {
                k - 1: v for k, v in self.encoded_cache.items() if k > 0
            }
    
    def get_recent_window(self, encoder: Optional[StateEncoder] = None) -> torch.Tensor:
        """Get recent observations as tensor."""
        recent = self.frames[-self.window_size:]
        
        if len(recent) == 0:
            # Return dummy if no observations
            return torch.zeros(1, 35)  # Default obs_dim
        
        # Convert to tensors
        tensors = [f.to_tensor() for f in recent]
        
        # Pad if needed
        while len(tensors) < self.window_size:
            tensors.insert(0, tensors[0].clone())  # Repeat first
        
        stacked = torch.stack(tensors)  # [window_size, obs_dim]
        
        if encoder is not None:
            stacked = torch.stack([encoder(t) for t in stacked])
        
        return stacked
    
    def get_trajectory_stats(self) -> Dict:
        """Get statistics about recent trajectory."""
        if len(self.frames) < 2:
            return {}
        
        recent = self.frames[-self.window_size:]
        
        losses = [f.loss for f in recent]
        accs = [f.accuracy for f in recent]
        
        return {
            'loss_trend': losses[-1] - losses[0] if len(losses) > 1 else 0,
            'acc_trend': accs[-1] - accs[0] if len(accs) > 1 else 0,
            'loss_mean': sum(losses) / len(losses),
            'acc_mean': sum(accs) / len(accs),
            'num_frames': len(self.frames),
        }
    
    def clear(self):
        """Clear buffer."""
        self.frames = []
        self.encoded_cache = {}
