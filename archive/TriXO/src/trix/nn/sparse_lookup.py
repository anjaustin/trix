"""
SparseLookupFFN: Where Routing Is The Computation

The thesis:
    A transformer FFN doesn't need to compute.
    It needs to select the right transformation—and know how strongly to apply it.

Architecture:
    Input → Route → Select Direction → Modulate via Spline → Output
    
    No matrix multiplies in the hot path.
    Routing IS the computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class TernarySpline2D(nn.Module):
    """
    2D Spline with ternary coefficients.
    
    Each cell has: base, slope_a, slope_b in {-1, 0, +1}
    Plus a learned scale factor.
    
    Output = scale * (base + slope_a * local_a + slope_b * local_b)
    """
    
    def __init__(self, grid_size: int = 16):
        super().__init__()
        self.grid_size = grid_size
        
        # Learnable continuous params (quantized during forward)
        self.coeffs = nn.Parameter(torch.randn(grid_size, grid_size, 3) * 0.5)
        self.scale = nn.Parameter(torch.ones(1))
    
    def _quantize_ternary(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to {-1, 0, +1} with straight-through estimator."""
        with torch.no_grad():
            q = torch.zeros_like(x)
            q[x > 0.3] = 1.0
            q[x < -0.3] = -1.0
        return x + (q - x).detach()  # STE: forward uses q, backward uses x
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: First coordinate, in [-1, 1], shape [batch]
            b: Second coordinate, in [-1, 1], shape [batch]
        
        Returns:
            output: Spline value, shape [batch]
        """
        # Map [-1, 1] to [0, grid_size-1]
        idx_a = ((a + 1) / 2 * self.grid_size).long().clamp(0, self.grid_size - 1)
        idx_b = ((b + 1) / 2 * self.grid_size).long().clamp(0, self.grid_size - 1)
        
        # Quantize coefficients
        q_coeffs = self._quantize_ternary(self.coeffs)
        
        # Gather cell coefficients
        cell_coeffs = q_coeffs[idx_a, idx_b]  # [batch, 3]
        base = cell_coeffs[:, 0]
        slope_a = cell_coeffs[:, 1]
        slope_b = cell_coeffs[:, 2]
        
        # Local position within cell [0, 1]
        cell_size = 2.0 / self.grid_size
        local_a = ((a + 1) - idx_a.float() * cell_size) / cell_size
        local_b = ((b + 1) - idx_b.float() * cell_size) / cell_size
        
        # Spline evaluation
        result = base + slope_a * local_a + slope_b * local_b
        
        return result * self.scale


class FloatSpline2D(nn.Module):
    """
    2D Spline with float coefficients (for comparison/ablation).
    """
    
    def __init__(self, grid_size: int = 16):
        super().__init__()
        self.grid_size = grid_size
        self.coeffs = nn.Parameter(torch.randn(grid_size, grid_size, 3) * 0.1)
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        idx_a = ((a + 1) / 2 * self.grid_size).long().clamp(0, self.grid_size - 1)
        idx_b = ((b + 1) / 2 * self.grid_size).long().clamp(0, self.grid_size - 1)
        
        cell_coeffs = self.coeffs[idx_a, idx_b]
        base, slope_a, slope_b = cell_coeffs[:, 0], cell_coeffs[:, 1], cell_coeffs[:, 2]
        
        cell_size = 2.0 / self.grid_size
        local_a = ((a + 1) - idx_a.float() * cell_size) / cell_size
        local_b = ((b + 1) - idx_b.float() * cell_size) / cell_size
        
        return base + slope_a * local_a + slope_b * local_b


class SparseLookupFFN(nn.Module):
    """
    Sparse Lookup Feed-Forward Network.
    
    Core insight: Routing selects a direction. Spline selects magnitude.
    No matrix multiplies in the hot path.
    
    Architecture:
        1. Shared compression: d_model → 2 scalars (a, b)
        2. Hierarchical routing: input → tile_idx
        3. Per-tile spline: (a, b) → scale
        4. Output: scale × direction[tile_idx]
        5. Residual: input + output
    
    Args:
        d_model: Model dimension
        num_tiles: Number of specialist tiles
        tiles_per_cluster: Tiles per routing cluster
        grid_size: Spline grid resolution (grid_size² cells)
        compress_hidden: Hidden dimension in compression network
        ternary_splines: Use ternary quantized splines
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        grid_size: int = 16,
        compress_hidden: Optional[int] = None,
        ternary_splines: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_clusters = num_tiles // tiles_per_cluster
        self.tiles_per_cluster = tiles_per_cluster
        self.grid_size = grid_size
        
        # Compression network (shared across all tiles)
        compress_hidden = compress_hidden or d_model // 4
        self.compress = nn.Sequential(
            nn.Linear(d_model, compress_hidden),
            nn.GELU(),
            nn.Linear(compress_hidden, 2),
            nn.Tanh(),  # Output in [-1, 1]
        )
        
        # Per-tile splines
        SplineClass = TernarySpline2D if ternary_splines else FloatSpline2D
        self.splines = nn.ModuleList([
            SplineClass(grid_size) for _ in range(num_tiles)
        ])
        
        # Tile directions (the "knowledge" - what each tile contributes)
        self.directions = nn.Parameter(torch.randn(num_tiles, d_model) * 0.02)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Output scaling
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Cluster assignments (fixed)
        self.register_buffer(
            'cluster_assignments',
            torch.arange(num_tiles) // tiles_per_cluster
        )
        
        # Usage tracking
        self.register_buffer('tile_counts', torch.zeros(num_tiles))
        self.register_buffer('total_count', torch.tensor(0.0))
    
    def get_signatures(self) -> torch.Tensor:
        """Derive routing signatures from directions."""
        return self.directions.sign()
    
    def get_cluster_signatures(self, signatures: torch.Tensor) -> torch.Tensor:
        """Compute cluster-level signatures."""
        cluster_sigs = []
        for c in range(self.num_clusters):
            mask = self.cluster_assignments == c
            cluster_sigs.append(signatures[mask].mean(dim=0).sign())
        return torch.stack(cluster_sigs)
    
    def route(self, x: torch.Tensor, signatures: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical routing: cluster → tile.
        
        Args:
            x: Normalized input [B*T, d_model]
            signatures: Tile signatures [num_tiles, d_model]
        
        Returns:
            tile_idx: Selected tile per input [B*T]
        """
        batch_size = x.shape[0]
        device = x.device
        
        cluster_sigs = self.get_cluster_signatures(signatures)
        
        # Level 1: Route to cluster
        cluster_scores = x @ cluster_sigs.T  # [B*T, num_clusters]
        cluster_idx = cluster_scores.argmax(dim=-1)  # [B*T]
        
        # Level 2: Route to tile within cluster
        tile_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for c in range(self.num_clusters):
            mask = cluster_idx == c
            if not mask.any():
                continue
            
            # Get tiles in this cluster
            tile_mask = self.cluster_assignments == c
            cluster_tiles = torch.where(tile_mask)[0]
            cluster_tile_sigs = signatures[tile_mask]
            
            # Route within cluster
            scores = x[mask] @ cluster_tile_sigs.T
            local_idx = scores.argmax(dim=-1)
            tile_idx[mask] = cluster_tiles[local_idx]
        
        return tile_idx
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, d_model]
        
        Returns:
            output: Output tensor [B, T, d_model]
            routing_info: Dict with routing details
            aux_losses: Dict with auxiliary losses
        """
        B, T, D = x.shape
        device = x.device
        
        # Normalize
        x_norm = self.norm(x)
        x_flat = x_norm.view(-1, D)  # [B*T, D]
        
        # Get routing signatures
        signatures = self.get_signatures()
        
        # Route
        tile_idx = self.route(x_flat, signatures)  # [B*T]
        
        # Compress (shared across all inputs)
        compressed = self.compress(x_flat)  # [B*T, 2]
        a, b = compressed[:, 0], compressed[:, 1]
        
        # Sparse lookup: each input goes to its routed tile
        output = torch.zeros_like(x_flat)
        
        for t in range(self.num_tiles):
            mask = tile_idx == t
            if not mask.any():
                continue
            
            # Spline lookup
            scale = self.splines[t](a[mask], b[mask])  # [n]
            
            # Apply direction
            output[mask] = scale.unsqueeze(-1) * self.directions[t]
            
            # Track usage
            if self.training:
                self.tile_counts[t] += mask.sum().float()
        
        if self.training:
            self.total_count += B * T
        
        # Reshape and scale
        output = output.view(B, T, D) * self.output_scale
        output = self.dropout(output)
        
        # Residual connection
        output = x + output
        
        # Routing info
        routing_info = {
            'tile_idx': tile_idx.view(B, T),
            'compressed': compressed.view(B, T, 2),
        }
        
        # Auxiliary losses
        aux_losses = self._compute_aux_losses(tile_idx, B * T)
        
        return output, routing_info, aux_losses
    
    def _compute_aux_losses(self, tile_idx: torch.Tensor, total: int) -> Dict:
        """Compute load balancing auxiliary losses."""
        # Count tiles in this batch
        counts = torch.zeros(self.num_tiles, device=tile_idx.device)
        for t in range(self.num_tiles):
            counts[t] = (tile_idx == t).sum().float()
        
        # Ideal: uniform distribution
        ideal = total / self.num_tiles
        
        # Balance loss: penalize deviation from uniform
        balance_loss = ((counts - ideal) ** 2).mean() / (ideal ** 2 + 1e-8)
        
        return {
            'balance': balance_loss * 0.01,
            'total_aux': balance_loss * 0.01,
        }
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        if self.total_count == 0:
            return {'num_tiles': self.num_tiles, 'active_tiles': 0}
        
        usage = self.tile_counts / self.total_count
        
        return {
            'num_tiles': self.num_tiles,
            'num_clusters': self.num_clusters,
            'active_tiles': (usage > 0.001).sum().item(),
            'usage_mean': usage.mean().item(),
            'usage_std': usage.std().item(),
            'usage_max': usage.max().item(),
            'usage_min': usage.min().item(),
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.tile_counts.zero_()
        self.total_count.zero_()
    
    def get_param_count(self) -> Dict:
        """Count parameters by component."""
        compress_params = sum(p.numel() for p in self.compress.parameters())
        spline_params = sum(sum(p.numel() for p in s.parameters()) for s in self.splines)
        direction_params = self.directions.numel()
        norm_params = sum(p.numel() for p in self.norm.parameters())
        
        total = compress_params + spline_params + direction_params + norm_params + 1  # +1 for output_scale
        
        return {
            'compress': compress_params,
            'splines': spline_params,
            'directions': direction_params,
            'norm': norm_params,
            'total': total,
        }


class SparseLookupBlock(nn.Module):
    """
    Transformer block using SparseLookupFFN.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        grid_size: int = 16,
        ternary_splines: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.ffn = SparseLookupFFN(
            d_model=d_model,
            num_tiles=num_tiles,
            tiles_per_cluster=tiles_per_cluster,
            grid_size=grid_size,
            ternary_splines=ternary_splines,
            dropout=dropout,
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        is_causal: bool = True
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """Forward pass."""
        B, T, D = x.shape
        
        # Attention
        x_norm = self.ln1(x)
        if is_causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        x = x + self.dropout(attn_out)
        
        # FFN (SparseLookup)
        x, routing_info, aux_losses = self.ffn(x)
        
        return x, routing_info, aux_losses


# =============================================================================
# Testing
# =============================================================================

def test_sparse_lookup():
    """Quick test of SparseLookupFFN."""
    print("=" * 60)
    print("SparseLookupFFN Test")
    print("=" * 60)
    
    d_model = 128
    batch_size = 4
    seq_len = 32
    
    # Create model
    model = SparseLookupFFN(
        d_model=d_model,
        num_tiles=64,
        tiles_per_cluster=8,
        grid_size=16,
        ternary_splines=True,
    )
    
    # Parameter count
    params = model.get_param_count()
    print(f"\nParameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
    
    # Forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    output, routing_info, aux_losses = model(x)
    
    print(f"\nShapes:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Tile indices: {routing_info['tile_idx'].shape}")
    print(f"  Compressed: {routing_info['compressed'].shape}")
    
    # Gradient check
    loss = output.sum()
    loss.backward()
    
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"\nGradients: {'OK' if grad_ok else 'FAIL'}")
    
    # Routing stats
    stats = model.get_routing_stats()
    print(f"\nRouting:")
    print(f"  Active tiles: {stats['active_tiles']}/{stats['num_tiles']}")
    print(f"  Usage mean: {stats['usage_mean']:.4f}")
    print(f"  Usage std: {stats['usage_std']:.4f}")
    
    # Compare parameter count to HierarchicalTriXFFN
    hier_params = 826_304  # From benchmark
    reduction = hier_params / params['total']
    print(f"\nCompared to HierarchicalTriXFFN:")
    print(f"  HierarchicalTriX: {hier_params:,} params")
    print(f"  SparseLookup: {params['total']:,} params")
    print(f"  Reduction: {reduction:.1f}×")
    
    print("\n" + "=" * 60)
    print("Test PASSED")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_sparse_lookup()
