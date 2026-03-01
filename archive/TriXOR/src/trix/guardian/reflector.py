"""
Superpositioned Reflector - Multi-Angle Self-View

"During training, we need to see ourselves from multiple angles to learn who we are.
Once trained, we carry those perspectives integrated within us.
The reflection becomes the self."

The XOR Reflector shows what changed.
The Superpositioned Reflector shows multiple views at once.
Together, they let the model navigate its own becoming.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class XORReflector(nn.Module):
    """
    XOR-based reflection showing what changed between states.
    
    XOR properties:
    - Self-inverse: a ⊕ b ⊕ b = a
    - Shows difference/delta
    - Creates orthogonality
    
    This is the retrospective function:
    - "Yeah, that was pretty good!" (small delta, good direction)
    - "That's where I missed it!" (large delta, wrong direction)
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
        Assess whether the change was "good" or "needs correction".
        
        This is NOT about right/wrong. It's about:
        - Is the change aligned with desired direction?
        - Is the entropy signaling a clear path?
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
            
            # "Yeah, that was pretty good!" vs "I got you next time!"
            if assessment['aligned']:
                assessment['message'] = "good_direction"
            else:
                assessment['message'] = "course_correct"
                assessment['correction_direction'] = target_direction - info['direction']
        
        return assessment


class SuperpositionedReflector(nn.Module):
    """
    Multi-angle self-view through N learned orthogonal bases.
    
    Instead of one mirror showing one reflection,
    we show the model N reflections simultaneously.
    
    The superposition allows:
    - Faster exploration (parallel perspectives)
    - Escape from local minima (one view might see the exit)
    - Richer gradients (multiple signals per step)
    - Reduced lr sensitivity (less commitment to single path)
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
    Reflection on the training process itself.
    
    This is the meta-level: not reflecting on representations,
    but reflecting on how training is going.
    
    "Yeah, that was pretty good!" - reinforce this learning direction
    "That's where I missed it! I got you next time!" - learn the anti-pattern
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
        """Get the Guardian Angel's message based on assessment."""
        idx = probs.argmax().item()
        
        if idx == 0:
            return "Yeah, look at 'em go! 🔥"
        elif idx == 1:
            return "Steady as she goes..."
        else:
            return "I got you next time! 💪"
