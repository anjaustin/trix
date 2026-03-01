"""
SparseLookupFFNv4: SpatioTemporal TriX

Combines three routing dimensions:
1. SPATIAL (GeometriX): Position on the manifold (B-spline spreading)
2. TEMPORAL: State-based routing (which branch of the manifold)
3. CONTENT: Standard TriX signature matching

The geometry:
- Spatial = where on the tube
- Temporal = which fork of the tube (state creates parallel paths)
- Content = what type of computation

For 6502 ADC: Carry flag creates two parallel tubes. C=0 and C=1 route differently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


def cubic_bspline(t: torch.Tensor) -> torch.Tensor:
    """Cubic B-spline kernel. C² continuous."""
    t = t.abs()
    result = torch.zeros_like(t)
    
    mask1 = t < 1
    result[mask1] = (2/3) - t[mask1]**2 + 0.5 * t[mask1]**3
    
    mask2 = (t >= 1) & (t < 2)
    result[mask2] = (1/6) * (2 - t[mask2])**3
    
    return result


class TernarySpline2D(nn.Module):
    """2D Spline with ternary coefficients."""
    
    def __init__(self, grid_size: int = 16):
        super().__init__()
        self.grid_size = grid_size
        self.coeffs = nn.Parameter(torch.randn(grid_size, grid_size, 3) * 0.5)
        self.scale = nn.Parameter(torch.ones(1))
    
    def _quantize_ternary(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = torch.zeros_like(x)
            q[x > 0.3] = 1.0
            q[x < -0.3] = -1.0
        return x + (q - x).detach()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        idx_a = ((a + 1) / 2 * self.grid_size).long().clamp(0, self.grid_size - 1)
        idx_b = ((b + 1) / 2 * self.grid_size).long().clamp(0, self.grid_size - 1)
        
        q_coeffs = self._quantize_ternary(self.coeffs)
        cell_coeffs = q_coeffs[idx_a, idx_b]
        
        base = cell_coeffs[:, 0]
        slope_a = cell_coeffs[:, 1]
        slope_b = cell_coeffs[:, 2]
        
        cell_size = 2.0 / self.grid_size
        local_a = ((a + 1) - idx_a.float() * cell_size) / cell_size
        local_b = ((b + 1) - idx_b.float() * cell_size) / cell_size
        
        result = base + slope_a * local_a + slope_b * local_b
        return result * self.scale


class SparseLookupFFNv4(nn.Module):
    """
    SpatioTemporal TriX: Three-way routing.
    
    Routes based on:
    1. Content (what) - signature matching
    2. Position (where) - B-spline spatial spreading  
    3. State (when/which branch) - temporal gating
    
    Args:
        d_model: Model dimension
        num_tiles: Number of expert tiles
        num_states: Number of discrete states (e.g., 2 for carry flag)
        state_dim: Dimension of continuous state vector
        max_seq_len: Maximum sequence length for positional routing
        position_spread: B-spline spreading width
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_tiles: int = 16,
        tiles_per_cluster: int = 4,
        num_states: int = 2,
        state_dim: int = 8,
        max_seq_len: int = 64,
        position_spread: float = 2.0,
        grid_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_clusters = num_tiles // tiles_per_cluster
        self.tiles_per_cluster = tiles_per_cluster
        self.num_states = num_states
        self.state_dim = state_dim
        self.max_seq_len = max_seq_len
        self.position_spread = position_spread
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Compression to spline coordinates
        compress_hidden = d_model // 4
        self.compress = nn.Sequential(
            nn.Linear(d_model, compress_hidden),
            nn.GELU(),
            nn.Linear(compress_hidden, 2),
            nn.Tanh(),
        )
        
        # Per-tile splines
        self.splines = nn.ModuleList([
            TernarySpline2D(grid_size) for _ in range(num_tiles)
        ])
        
        # Tile directions (content signatures)
        self.directions = nn.Parameter(torch.randn(num_tiles, d_model) * 0.02)
        
        # === SPATIAL: Position-based routing ===
        self.register_buffer(
            'tile_positions',
            torch.linspace(0, max_seq_len, num_tiles)
        )
        
        # === TEMPORAL: State-based routing ===
        # Each tile has preferred states (learns which states it handles)
        self.state_signatures = nn.Parameter(torch.randn(num_tiles, state_dim) * 0.1)
        
        # State encoder: discrete state → continuous state vector
        self.state_encoder = nn.Embedding(num_states, state_dim)
        
        # State-conditioned tile modulation
        # When in state s, modulate tile output
        self.state_modulation = nn.Parameter(torch.ones(num_states, num_tiles))
        
        # Cluster assignments
        self.register_buffer(
            'cluster_assignments',
            torch.arange(num_tiles) // tiles_per_cluster
        )
        
        # Output
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.dropout = nn.Dropout(dropout)
        
        # Stats
        self.register_buffer('tile_state_counts', torch.zeros(num_tiles, num_states))
        self.register_buffer('total_count', torch.tensor(0.0))
    
    def get_signatures(self) -> torch.Tensor:
        return self.directions.sign()
    
    def spatiotemporal_routing(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        states: torch.Tensor,
        signatures: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Three-way routing: content × position × state.
        
        Args:
            x: [B*T, D] input
            positions: [B*T] position indices
            states: [B*T] discrete state indices
            signatures: [num_tiles, D] ternary signatures
        
        Returns:
            tile_idx: [B*T] winning tile
            tile_weights: [B*T] confidence
        """
        B_T, D = x.shape
        device = x.device
        
        # === CONTENT SCORES ===
        content_scores = x @ signatures.T  # [B*T, num_tiles]
        
        # === SPATIAL SCORES (B-spline) ===
        pos_normalized = positions / self.max_seq_len * self.num_tiles
        tile_centers = torch.arange(self.num_tiles, device=device).float()
        pos_diff = pos_normalized.unsqueeze(-1) - tile_centers.unsqueeze(0)
        spatial_scores = cubic_bspline(pos_diff / self.position_spread)
        
        # === TEMPORAL SCORES (state matching) ===
        state_vecs = self.state_encoder(states)  # [B*T, state_dim]
        temporal_scores = state_vecs @ self.state_signatures.T  # [B*T, num_tiles]
        temporal_scores = torch.sigmoid(temporal_scores)  # 0-1 range
        
        # === COMBINED ROUTING ===
        # All three must align for strong routing
        combined_scores = content_scores * spatial_scores * temporal_scores
        
        # Softmax for differentiability
        routing_probs = F.softmax(combined_scores * 5.0, dim=-1)
        
        # Hard routing (argmax) with soft gradients
        tile_idx = combined_scores.argmax(dim=-1)
        tile_weights = routing_probs.gather(1, tile_idx.unsqueeze(-1)).squeeze(-1)
        
        return tile_idx, tile_weights
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Forward pass with spatiotemporal routing.
        
        Args:
            x: [B, T, d_model]
            positions: [B, T] position indices (default: 0, 1, 2, ...)
            states: [B, T] discrete state indices (default: all 0)
        """
        B, T, D = x.shape
        device = x.device
        
        # Defaults
        if positions is None:
            positions = torch.arange(T, device=device).float().unsqueeze(0).expand(B, -1)
        if states is None:
            states = torch.zeros(B, T, dtype=torch.long, device=device)
        
        # Flatten
        x_norm = self.norm(x)
        x_flat = x_norm.reshape(-1, D)
        pos_flat = positions.reshape(-1)
        state_flat = states.reshape(-1)
        
        # Get signatures
        signatures = self.get_signatures()
        
        # Three-way routing
        tile_idx, tile_weights = self.spatiotemporal_routing(
            x_flat, pos_flat, state_flat, signatures
        )
        
        # Compress to spline coordinates
        compressed = self.compress(x_flat)
        a, b = compressed[:, 0], compressed[:, 1]
        
        # Compute output per tile
        output = torch.zeros_like(x_flat)
        
        for t in range(self.num_tiles):
            mask = tile_idx == t
            if not mask.any():
                continue
            
            # Spline lookup
            scale = self.splines[t](a[mask], b[mask])
            
            # State modulation: adjust contribution based on state
            state_mod = self.state_modulation[state_flat[mask], t]
            
            # Output
            contribution = (scale * state_mod).unsqueeze(-1) * self.directions[t]
            output[mask] = contribution
            
            # Track stats
            if self.training:
                for s in range(self.num_states):
                    state_mask = state_flat[mask] == s
                    self.tile_state_counts[t, s] += state_mask.sum().float()
        
        if self.training:
            self.total_count += B * T
        
        # Reshape and scale
        output = output.reshape(B, T, D) * self.output_scale
        output = self.dropout(output)
        
        # Residual
        output = x + output
        
        # Info
        routing_info = {
            'tile_idx': tile_idx.reshape(B, T),
            'tile_weights': tile_weights.reshape(B, T),
            'states': states,
        }
        
        # Aux losses
        aux_losses = self._compute_aux_losses(tile_idx, state_flat, B * T)
        
        return output, routing_info, aux_losses
    
    def _compute_aux_losses(self, tile_idx: torch.Tensor, states: torch.Tensor, total: int) -> Dict:
        """Compute auxiliary losses including state-aware balancing."""
        device = tile_idx.device
        
        # Per-state tile balance
        balance_loss = torch.tensor(0.0, device=device)
        
        for s in range(self.num_states):
            state_mask = states == s
            if state_mask.sum() == 0:
                continue
            
            counts = torch.zeros(self.num_tiles, device=device)
            for t in range(self.num_tiles):
                counts[t] = ((tile_idx == t) & state_mask).sum().float()
            
            state_total = state_mask.sum().float()
            ideal = state_total / self.num_tiles
            balance_loss += ((counts - ideal) ** 2).mean() / (ideal ** 2 + 1e-8)
        
        balance_loss = balance_loss / self.num_states
        
        total_aux = balance_loss * 0.01
        
        return {
            'balance': balance_loss * 0.01,
            'total_aux': total_aux,
        }
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics including state breakdown."""
        if self.total_count == 0:
            return {'num_tiles': self.num_tiles, 'active_tiles': 0}
        
        usage = self.tile_state_counts.sum(dim=1) / self.total_count
        
        stats = {
            'num_tiles': self.num_tiles,
            'num_states': self.num_states,
            'active_tiles': (usage > 0.001).sum().item(),
            'usage_mean': usage.mean().item(),
            'usage_std': usage.std().item(),
        }
        
        # Per-state tile preferences
        state_prefs = {}
        for s in range(self.num_states):
            state_counts = self.tile_state_counts[:, s]
            if state_counts.sum() > 0:
                top_tile = state_counts.argmax().item()
                state_prefs[f'state_{s}_top_tile'] = top_tile
        stats['state_preferences'] = state_prefs
        
        return stats
    
    def reset_stats(self):
        self.tile_state_counts.zero_()
        self.total_count.zero_()
    
    def get_param_count(self) -> Dict:
        compress = sum(p.numel() for p in self.compress.parameters())
        splines = sum(sum(p.numel() for p in s.parameters()) for s in self.splines)
        directions = self.directions.numel()
        state_enc = sum(p.numel() for p in self.state_encoder.parameters())
        state_sig = self.state_signatures.numel()
        state_mod = self.state_modulation.numel()
        norm = sum(p.numel() for p in self.norm.parameters())
        
        total = compress + splines + directions + state_enc + state_sig + state_mod + norm + 1
        
        return {
            'compress': compress,
            'splines': splines,
            'directions': directions,
            'state_encoder': state_enc,
            'state_signatures': state_sig,
            'state_modulation': state_mod,
            'norm': norm,
            'total': total,
        }


def test_v4():
    """Quick test of SparseLookupFFNv4."""
    print("=" * 60)
    print("SparseLookupFFNv4 (SpatioTemporal TriX) Test")
    print("=" * 60)
    
    d_model = 128
    batch_size = 4
    seq_len = 8
    
    model = SparseLookupFFNv4(
        d_model=d_model,
        num_tiles=16,
        tiles_per_cluster=4,
        num_states=2,  # e.g., carry=0, carry=1
        state_dim=8,
        max_seq_len=64,
    )
    
    params = model.get_param_count()
    print(f"\nParameters: {params['total']:,}")
    
    # Test with states
    x = torch.randn(batch_size, seq_len, d_model)
    positions = torch.arange(seq_len).float().unsqueeze(0).expand(batch_size, -1)
    states = torch.randint(0, 2, (batch_size, seq_len))  # Random carry flags
    
    output, info, aux = model(x, positions, states)
    
    print(f"\nShapes:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  States: {states.shape}")
    
    print(f"\nRouting (first sequence):")
    print(f"  States: {states[0].tolist()}")
    print(f"  Tiles:  {info['tile_idx'][0].tolist()}")
    
    # Check gradient flow
    loss = output.sum() + aux['total_aux']
    loss.backward()
    
    grad_ok = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"\nGradients: {grad_ok}/{total_params} params have grads")
    
    print("\n" + "=" * 60)
    print("Test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_v4()
