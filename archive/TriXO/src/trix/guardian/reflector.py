"""
Reflector Modules - State Change Analysis

Two complementary reflection mechanisms:

- XORReflector: Analyzes what changed between states (delta analysis)
- SuperpositionedReflector: Projects state through multiple bases (multi-view)

These enable the observer to understand training trajectory and
assess whether changes are moving in productive directions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class XORReflector(nn.Module):
    """
    XOR-based reflection showing what changed between states.

    Computes a learned transformation of the delta between current and
    previous state, providing:

    - Delta magnitude (how much changed)
    - Delta direction (unit vector of change)
    - Significance detection (whether change exceeds threshold)

    Used for trajectory assessment during training.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Learnable XOR-like mixing (from our proven XOR mixer)
        self.mix_weight = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.mix_bias = nn.Parameter(torch.zeros(d_model))
        
        # Threshold for "significant change" detection
        self.register_buffer('change_threshold', torch.tensor(0.1))
        
    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute XOR-like reflection between current and previous state.
        
        Returns:
            reflection: the XOR-mixed delta
            info: dictionary with analysis
        """
        # Ternary-like representation
        current_tern = torch.tanh(current)
        previous_tern = torch.tanh(previous)
        
        # XOR-like operation: what's different?
        delta = current_tern - previous_tern
        
        # Apply learned mixing
        reflection = torch.matmul(delta, self.mix_weight) + self.mix_bias
        
        # Analysis
        delta_magnitude = torch.norm(delta, dim=-1)
        significant_change = delta_magnitude > self.change_threshold
        
        info = {
            'delta': delta,
            'magnitude': delta_magnitude,
            'significant': significant_change,
            'direction': F.normalize(delta, dim=-1),  # Unit vector of change
        }
        
        return reflection, info
    
    def assess_trajectory(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        target_direction: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Assess whether the state change is moving in a good direction.

        If target_direction is provided, computes alignment between
        actual change direction and desired direction.

        Returns dict with magnitude, direction, and alignment info.
        """
        _, info = self.forward(current, previous)

        assessment = {
            'change_magnitude': info['magnitude'].mean().item(),
            'change_direction': info['direction'],
        }

        if target_direction is not None:
            # How aligned is our change with target direction?
            alignment = F.cosine_similarity(
                info['direction'],
                target_direction,
                dim=-1
            )
            assessment['alignment'] = alignment.mean().item()
            assessment['aligned'] = alignment.mean().item() > 0.5

            if assessment['aligned']:
                assessment['message'] = "good_direction"
            else:
                assessment['message'] = "course_correct"
                assessment['correction_direction'] = target_direction - info['direction']

        return assessment


class SuperpositionedReflector(nn.Module):
    """
    Multi-view projection through N learned orthogonal bases.

    Projects the input through multiple learned transformation matrices,
    then combines the results with learnable weights. This provides:

    - Multiple perspectives on the same representation
    - Richer gradient signal during training
    - Regularization effect from orthogonality constraint
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_bases: int = 4,
        learnable_combination: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_bases = num_bases
        
        # Learned orthogonal bases (initialized near-orthogonal)
        bases = torch.randn(num_bases, d_model, d_model) * 0.1
        # Encourage orthogonality via initialization
        for i in range(num_bases):
            bases[i] = torch.linalg.qr(bases[i])[0]
        self.bases = nn.Parameter(bases * 0.5)  # Scale down for stability
        
        # Bias per basis
        self.biases = nn.Parameter(torch.zeros(num_bases, d_model))
        
        # Learnable combination weights (or equal weighting)
        if learnable_combination:
            self.combination_weights = nn.Parameter(torch.ones(num_bases) / num_bases)
        else:
            self.register_buffer('combination_weights', torch.ones(num_bases) / num_bases)
        
        # XOR reflector for temporal analysis
        self.xor_reflector = XORReflector(d_model)
        
    def reflect_single_basis(self, x: torch.Tensor, basis_idx: int) -> torch.Tensor:
        """Reflect input through a single basis."""
        basis = self.bases[basis_idx]  # [d_model, d_model]
        bias = self.biases[basis_idx]  # [d_model]
        
        # Ternary-like activation before reflection
        x_tern = torch.tanh(x)
        
        # Reflect through basis
        reflected = torch.matmul(x_tern, basis) + bias
        
        return reflected
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Show input N views of itself simultaneously.
        
        Returns:
            superposed: the superposition of all reflections (added to input as residual)
            info: per-basis information
        """
        reflections = []
        
        for i in range(self.num_bases):
            reflected = self.reflect_single_basis(x, i)
            reflections.append(reflected)
        
        # Stack reflections: [num_bases, batch, ..., d_model]
        reflections_stack = torch.stack(reflections, dim=0)
        
        # Normalize combination weights
        weights = F.softmax(self.combination_weights, dim=0)
        
        # Weighted superposition
        # weights: [num_bases] -> [num_bases, 1, 1, ...] for broadcasting
        weight_shape = [self.num_bases] + [1] * (reflections_stack.dim() - 1)
        weighted = reflections_stack * weights.view(*weight_shape)
        superposed = weighted.sum(dim=0)  # [batch, ..., d_model]
        
        # Add as residual (the reflection augments, doesn't replace)
        output = x + superposed
        
        info = {
            'reflections': reflections_stack,
            'combination_weights': weights,
            'per_basis_norms': [r.norm().item() for r in reflections],
            'superposition_norm': superposed.norm().item(),
        }
        
        return output, info
    
    def get_orthogonality_loss(self) -> torch.Tensor:
        """
        Regularization loss to encourage basis orthogonality.
        
        Orthogonal bases provide maximally different perspectives.
        """
        loss = 0.0
        
        for i in range(self.num_bases):
            for j in range(i + 1, self.num_bases):
                # Frobenius inner product between bases
                inner = torch.sum(self.bases[i] * self.bases[j])
                loss = loss + inner ** 2
        
        return loss
    
    def analyze_diversity(self) -> dict:
        """Analyze how diverse the reflections are."""
        # Compute pairwise similarities between bases
        similarities = []
        for i in range(self.num_bases):
            for j in range(i + 1, self.num_bases):
                sim = F.cosine_similarity(
                    self.bases[i].flatten().unsqueeze(0),
                    self.bases[j].flatten().unsqueeze(0)
                ).item()
                similarities.append(sim)
        
        return {
            'mean_similarity': sum(similarities) / len(similarities) if similarities else 0,
            'max_similarity': max(similarities) if similarities else 0,
            'min_similarity': min(similarities) if similarities else 0,
            'num_bases': self.num_bases,
            'combination_weights': F.softmax(self.combination_weights, dim=0).tolist(),
        }


class TrainingManifoldReflector(nn.Module):
    """
    Meta-level reflection on the training trajectory.

    Analyzes training state history to assess whether the optimization
    is progressing well, neutral, or needs correction. Generates
    correction direction vectors when intervention is warranted.

    Architecture:
    - LSTM encoder for temporal state history
    - 3-way classifier: good / neutral / needs_correction
    - Correction generator for adjustment vectors
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        
        # Encode training state history
        self.state_encoder = nn.LSTM(
            state_dim, hidden_dim, 
            num_layers=2, 
            batch_first=True
        )
        
        # Assess trajectory quality
        self.trajectory_assessor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # [good, neutral, needs_correction]
        )
        
        # Generate correction direction if needed
        self.correction_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
    def forward(
        self, 
        state_history: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Reflect on training trajectory.
        
        state_history: [seq_len, state_dim] - recent training states
        
        Returns:
            assessment: trajectory quality scores
            info: including correction direction if needed
        """
        # Encode history
        _, (h, _) = self.state_encoder(state_history.unsqueeze(0))
        context = h[-1, 0]  # [hidden_dim]
        
        # Assess trajectory
        assessment = self.trajectory_assessor(context)
        assessment_probs = F.softmax(assessment, dim=-1)
        
        # Generate correction direction
        correction = self.correction_generator(context)
        
        info = {
            'good_prob': assessment_probs[0].item(),
            'neutral_prob': assessment_probs[1].item(),
            'needs_correction_prob': assessment_probs[2].item(),
            'correction_direction': correction,
            'message': self._get_message(assessment_probs),
        }
        
        return assessment_probs, info
    
    def _get_message(self, probs: torch.Tensor) -> str:
        """Get status message based on trajectory assessment."""
        idx = probs.argmax().item()

        if idx == 0:
            return "good_trajectory"
        elif idx == 1:
            return "neutral_trajectory"
        else:
            return "needs_correction"
