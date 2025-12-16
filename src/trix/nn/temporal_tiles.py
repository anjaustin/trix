"""
Temporal Tile Layer - Mesa 4

Extends TriX routing into time. Routes based on (input, state) rather than
just input, enabling learned state transitions without attention.

Core insight: Not "which tokens attend to which" but "which dynamics apply now."

Usage:
    layer = TemporalTileLayer(d_model=64, d_state=16, num_tiles=8)
    state = layer.init_state(batch_size=2, device='cuda')
    
    for t in range(seq_len):
        output, state, info = layer(x[:, t], state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class TemporalRoutingInfo:
    """Information about temporal routing decisions."""
    tile_idx: torch.Tensor      # Selected tile indices
    scores: torch.Tensor        # Routing scores
    state_contribution: float   # How much state affected routing


class TemporalTileLayer(nn.Module):
    """
    A tile layer that routes based on (input, state) and updates state.
    
    Each tile is a learned state transition function:
        (x_t, state_{t-1}) → (y_t, state_t)
    
    This enables temporal patterns without attention.
    
    Args:
        d_model: Input/output dimension
        d_state: State vector dimension (small: 16-64)
        num_tiles: Number of temporal tiles
        state_init: How to initialize state ('zero', 'learned')
        routing_temp: Temperature for routing (lower = harder)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        num_tiles: int = 8,
        state_init: str = 'zero',
        routing_temp: float = 1.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.num_tiles = num_tiles
        self.state_init = state_init
        self.routing_temp = routing_temp
        
        # Combined dimension for routing
        d_combined = d_model + d_state
        
        # Temporal signatures: route based on (input, state)
        self.signatures = nn.Parameter(torch.randn(num_tiles, d_combined))
        nn.init.orthogonal_(self.signatures)
        
        # Each tile has a state update and output transform
        # Using linear layers for simplicity; could be more complex
        self.state_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_combined, d_state * 2),
                nn.GELU(),
                nn.Linear(d_state * 2, d_state),
            ) for _ in range(num_tiles)
        ])
        
        self.output_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_combined, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            ) for _ in range(num_tiles)
        ])
        
        # Learned initial state (optional)
        if state_init == 'learned':
            self.init_state_param = nn.Parameter(torch.zeros(d_state))
        
        # For claim tracking (which tiles handle which regimes)
        self.register_buffer('transition_counts', torch.zeros(num_tiles, num_tiles))
        self.register_buffer('tile_counts', torch.zeros(num_tiles))
        
        # Layer norm for stability
        self.input_norm = nn.LayerNorm(d_model)
        self.state_norm = nn.LayerNorm(d_state)
    
    def init_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Initialize state for a new sequence."""
        if device is None:
            device = self.signatures.device
            
        if self.state_init == 'learned':
            return self.init_state_param.unsqueeze(0).expand(batch_size, -1).clone()
        else:
            return torch.zeros(batch_size, self.d_state, device=device)
    
    def route(self, x: torch.Tensor, state: torch.Tensor, hard: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route based on (input, state).
        
        Returns:
            tile_idx: Selected tile indices (batch,)
            scores: Full score matrix (batch, num_tiles)
            weights: Soft weights for mixing (batch, num_tiles)
        """
        # Normalize inputs
        x_norm = self.input_norm(x)
        state_norm = self.state_norm(state)
        
        # Concatenate for routing
        combined = torch.cat([x_norm, state_norm], dim=-1)
        
        # Compute scores
        scores = combined @ self.signatures.T / self.routing_temp
        
        # Soft weights (always differentiable)
        weights = F.softmax(scores, dim=-1)
        
        # Hard routing (argmax) with straight-through estimator
        tile_idx = scores.argmax(dim=-1)
        
        if hard and self.training:
            # Straight-through: use hard indices but gradient flows through soft weights
            one_hot = F.one_hot(tile_idx, self.num_tiles).float()
            # Straight-through trick: forward uses one_hot, backward uses weights
            weights = one_hot - weights.detach() + weights
        
        return tile_idx, scores, weights
    
    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        prev_tile: Optional[torch.Tensor] = None,
        track_transitions: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass for one timestep.
        
        Args:
            x: Input tensor (batch, d_model)
            state: Previous state (batch, d_state)
            prev_tile: Previous tile indices for transition tracking (batch,)
            track_transitions: Whether to update transition counts
        
        Returns:
            output: Transformed input (batch, d_model)
            new_state: Updated state (batch, d_state)
            info: Routing information dict
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Route to tile
        tile_idx, scores, weights = self.route(x, state)
        
        # Compute state contribution to routing
        x_only_scores = self.input_norm(x) @ self.signatures[:, :self.d_model].T
        state_contribution = 1.0 - F.cosine_similarity(
            scores, x_only_scores, dim=-1
        ).mean().item()
        
        # Normalize inputs for tile computation
        x_norm = self.input_norm(x)
        state_norm = self.state_norm(state)
        combined = torch.cat([x_norm, state_norm], dim=-1)
        
        # Execute all tiles and mix with weights (differentiable)
        all_outputs = torch.stack([
            self.output_transforms[i](combined) for i in range(self.num_tiles)
        ], dim=1)  # (batch, num_tiles, d_model)
        
        all_states = torch.stack([
            self.state_updates[i](combined) for i in range(self.num_tiles)
        ], dim=1)  # (batch, num_tiles, d_state)
        
        # Weighted combination (during training, weights have STE gradient)
        output = (weights.unsqueeze(-1) * all_outputs).sum(dim=1)  # (batch, d_model)
        new_state = (weights.unsqueeze(-1) * all_states).sum(dim=1)  # (batch, d_state)
        
        # Track transitions
        if track_transitions and prev_tile is not None:
            with torch.no_grad():
                for b in range(batch_size):
                    prev_t = prev_tile[b].item()
                    curr_t = tile_idx[b].item()
                    self.transition_counts[prev_t, curr_t] += 1
                    self.tile_counts[curr_t] += 1
        elif track_transitions:
            with torch.no_grad():
                for b in range(batch_size):
                    self.tile_counts[tile_idx[b].item()] += 1
        
        # Residual connection for output
        output = x + output
        
        info = {
            'tile_idx': tile_idx,
            'scores': scores,
            'state_contribution': state_contribution,
        }
        
        return output, new_state, info
    
    def forward_sequence(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Process a full sequence.
        
        Args:
            x: Input sequence (batch, seq_len, d_model)
            state: Initial state (batch, d_state) or None
        
        Returns:
            output: Output sequence (batch, seq_len, d_model)
            final_state: Final state (batch, d_state)
            infos: List of routing info dicts per timestep
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if state is None:
            state = self.init_state(batch_size, device)
        
        outputs = []
        infos = []
        prev_tile = None
        
        for t in range(seq_len):
            out, state, info = self.forward(x[:, t], state, prev_tile)
            outputs.append(out)
            infos.append(info)
            prev_tile = info['tile_idx']
        
        output = torch.stack(outputs, dim=1)
        return output, state, infos
    
    def get_transition_matrix(self, normalize: bool = True) -> torch.Tensor:
        """Get the learned transition matrix between tiles."""
        matrix = self.transition_counts.clone()
        if normalize:
            row_sums = matrix.sum(dim=1, keepdim=True)
            row_sums = row_sums.clamp(min=1)  # Avoid division by zero
            matrix = matrix / row_sums
        return matrix
    
    def get_tile_usage(self, normalize: bool = True) -> torch.Tensor:
        """Get tile usage distribution."""
        usage = self.tile_counts.clone()
        if normalize:
            usage = usage / usage.sum().clamp(min=1)
        return usage
    
    def reset_tracking(self):
        """Reset transition and usage tracking."""
        self.transition_counts.zero_()
        self.tile_counts.zero_()
    
    def get_regime_analysis(self) -> Dict:
        """Analyze the learned regime structure."""
        trans = self.get_transition_matrix(normalize=True)
        usage = self.get_tile_usage(normalize=True)
        
        # Find stable regimes (high self-transition probability)
        self_trans = trans.diag()
        stable_tiles = (self_trans > 0.5).nonzero().squeeze(-1).tolist()
        
        # Find transition hubs (high outgoing diversity)
        trans_entropy = -(trans * (trans + 1e-10).log()).sum(dim=1)
        hub_tiles = (trans_entropy > 1.0).nonzero().squeeze(-1).tolist()
        
        return {
            'transition_matrix': trans,
            'usage': usage,
            'stable_tiles': stable_tiles,
            'hub_tiles': hub_tiles,
            'self_transition_probs': self_trans,
        }


class TemporalTileStack(nn.Module):
    """
    Stack of temporal tile layers for deeper temporal processing.
    
    Each layer can have different state dimensions and tile counts.
    """
    
    def __init__(
        self,
        d_model: int,
        layers: List[Tuple[int, int]],  # [(d_state, num_tiles), ...]
        routing_temp: float = 1.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.layers = nn.ModuleList([
            TemporalTileLayer(
                d_model=d_model,
                d_state=d_state,
                num_tiles=num_tiles,
                routing_temp=routing_temp,
            )
            for d_state, num_tiles in layers
        ])
    
    def init_states(self, batch_size: int, device: torch.device = None) -> List[torch.Tensor]:
        """Initialize states for all layers."""
        return [layer.init_state(batch_size, device) for layer in self.layers]
    
    def forward_sequence(
        self,
        x: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[List[Dict]]]:
        """
        Process sequence through all layers.
        
        Returns:
            output: Final output sequence
            final_states: List of final states per layer
            all_infos: Nested list of routing infos [layer][timestep]
        """
        batch_size = x.shape[0]
        device = x.device
        
        if states is None:
            states = self.init_states(batch_size, device)
        
        all_infos = []
        current = x
        final_states = []
        
        for layer, state in zip(self.layers, states):
            current, final_state, infos = layer.forward_sequence(current, state)
            final_states.append(final_state)
            all_infos.append(infos)
        
        return current, final_states, all_infos
