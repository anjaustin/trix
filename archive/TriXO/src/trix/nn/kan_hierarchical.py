"""
KAN-Hierarchical TriX: PQH Architecture Implementation

Replaces MLP tiles with SharedAdditiveKAN tiles.
Keeps hierarchical routing for O(sqrt(n)) complexity.
Adds VGem's Residual Bus for stacking.

This is the integration of:
- Track B (Additive KAN)
- HierarchicalTriXFFN (routing)
- PQH (Pseudo-Quasi-Holographic) architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math

from .additive_kan import Spline1D, SharedAdditiveKAN


class KANTile(nn.Module):
    """
    A KAN-based tile for PQH architecture.
    
    KEY INSIGHT: Each tile handles a SLICE of dimensions, not all of them.
    This is what makes it sparse.
    
    Architecture:
    - Input: d_model dimensions
    - Active slice: d_slice dimensions (d_model / num_tiles)
    - Each tile only transforms its slice
    - Splines per dimension: grid_size params each
    
    Memory per tile: d_slice × grid_size × 2 params
    With d_model=256, 64 tiles, grid=16: 4 × 16 × 2 = 128 params = 32 bytes (2-bit)
    """
    
    def __init__(
        self,
        d_model: int,
        d_slice: int = None,
        grid_size: int = 16,
        tile_id: int = 0,
        num_tiles: int = 64,
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_slice = d_slice or max(d_model // num_tiles, 4)
        self.grid_size = grid_size
        self.tile_id = tile_id
        self.use_residual = use_residual
        
        # Which dimensions this tile handles
        # Each tile owns a slice of the full dimension space
        start_dim = (tile_id * self.d_slice) % d_model
        self.register_buffer('dim_indices', torch.arange(start_dim, start_dim + self.d_slice) % d_model)
        
        # Per-dimension splines (no mixing matrix!)
        # Each spline: base + slope per grid cell = 2 × grid_size params
        self.spline_bases = nn.Parameter(torch.zeros(self.d_slice, grid_size))
        self.spline_slopes = nn.Parameter(torch.ones(self.d_slice, grid_size) * 0.1)
        
        # Learnable output scale
        self.output_scale = nn.Parameter(torch.ones(1))
        
        # Usage tracking
        self.register_buffer('activation_count', torch.tensor(0.0))
        self.register_buffer('total_count', torch.tensor(0.0))
        
    def get_signature(self) -> torch.Tensor:
        """
        Extract signature from spline parameters.
        
        The signature indicates which input patterns this tile prefers.
        Derived from the mean slopes (positive = wants high, negative = wants low).
        """
        with torch.no_grad():
            # Full signature (d_model), mostly zeros except for our slice
            signature = torch.zeros(self.d_model, device=self.spline_slopes.device)
            signature[self.dim_indices] = self.spline_slopes.mean(dim=1).sign()
        return signature
    
    def _evaluate_splines(self, x_slice: torch.Tensor) -> torch.Tensor:
        """
        Evaluate 1D splines on the slice.
        
        x_slice: [batch, d_slice]
        returns: [batch, d_slice]
        """
        # Normalize to [0, 1) for grid indexing
        x_norm = (x_slice - x_slice.min()) / (x_slice.max() - x_slice.min() + 1e-8)
        x_norm = x_norm.clamp(0, 1 - 1e-6)
        
        # Grid indices
        idx = (x_norm * self.grid_size).long()
        idx = idx.clamp(0, self.grid_size - 1)  # [batch, d_slice]
        
        # Gather coefficients for each dimension
        # bases: [d_slice, grid_size], idx: [batch, d_slice]
        batch_size = x_slice.shape[0]
        
        # Reshape for gather
        bases = self.spline_bases.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, d_slice, grid]
        slopes = self.spline_slopes.unsqueeze(0).expand(batch_size, -1, -1)
        
        idx_expanded = idx.unsqueeze(-1)  # [batch, d_slice, 1]
        
        b = bases.gather(2, idx_expanded).squeeze(-1)  # [batch, d_slice]
        s = slopes.gather(2, idx_expanded).squeeze(-1)
        
        # Local offset within cell
        cell_size = 1.0 / self.grid_size
        x_local = (x_norm - idx.float() * cell_size) / cell_size
        
        # Evaluate: y = base + slope * x_local
        return b + s * x_local
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparse slice computation.
        
        Only modifies the dimensions this tile owns.
        VGem's Residual Bus: Output = Input + Delta
        """
        # Extract our slice
        x_slice = x[..., self.dim_indices]  # [batch, d_slice]
        
        # Compute through splines
        delta = self._evaluate_splines(x_slice) * self.output_scale
        
        # Build output (mostly passthrough, modify our slice)
        if self.use_residual:
            out = x.clone()
            out[..., self.dim_indices] = out[..., self.dim_indices] + delta
            return out
        else:
            out = torch.zeros_like(x)
            out[..., self.dim_indices] = delta
            return out
    
    def update_usage(self, count: int, total: int):
        """Track activation frequency."""
        self.activation_count += count
        self.total_count += total
    
    @property
    def usage_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.activation_count / self.total_count).item()
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def memory_bytes(self, bits: int = 32) -> int:
        """Memory usage in bytes."""
        return self.num_parameters() * bits // 8


class HierarchicalKANFFN(nn.Module):
    """
    Hierarchical KAN FFN - PQH Architecture.
    
    2-level routing with KAN tiles:
      Level 1: Route to cluster (coarse) - O(sqrt(n))
      Level 2: Route to tile within cluster (fine) - O(sqrt(n))
      Level 3: KAN spline lookup (implicit) - O(grid_size)
    
    Total routing: O(sqrt(n) + grid_size) ≈ O(sqrt(n))
    
    Memory per tile (2-bit): ~1.5KB
    Memory for 64 tiles: ~100KB
    """
    
    def __init__(
        self,
        d_model: int,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        grid_size: int = 16,
        dropout: float = 0.1,
        balance_weight: float = 0.01,
        ema_decay: float = 0.999,
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.tiles_per_cluster = tiles_per_cluster
        self.num_clusters = num_tiles // tiles_per_cluster
        self.grid_size = grid_size
        self.balance_weight = balance_weight
        self.ema_decay = ema_decay
        self.use_residual = use_residual
        
        # Input normalization for stable routing
        self.input_norm = nn.LayerNorm(d_model)
        
        # KAN tiles - each handles a slice of dimensions
        self.tiles = nn.ModuleList([
            KANTile(
                d_model=d_model,
                grid_size=grid_size,
                tile_id=i,
                num_tiles=num_tiles,
                use_residual=False,  # Tile residual off, global residual on
            )
            for i in range(num_tiles)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # EMA signatures for stable routing
        self.register_buffer('ema_tile_signatures', None)
        self.register_buffer('cluster_signatures', None)
        self.register_buffer('cluster_assignments', None)
        
        self._hierarchy_built = False
        
    def _get_current_signatures(self) -> torch.Tensor:
        """Get current signatures from all tiles."""
        return torch.stack([tile.get_signature() for tile in self.tiles])
    
    def _update_ema_signatures(self):
        """Update EMA of tile signatures."""
        current = self._get_current_signatures()
        
        if self.ema_tile_signatures is None:
            self.ema_tile_signatures = current.clone()
        else:
            self.ema_tile_signatures = (
                self.ema_decay * self.ema_tile_signatures +
                (1 - self.ema_decay) * current
            )
    
    def _build_hierarchy(self):
        """Build cluster hierarchy from tile signatures."""
        self._update_ema_signatures()
        
        signatures = self.ema_tile_signatures.sign()
        
        # Simple clustering: group consecutive tiles
        # (In production, use k-means on signatures)
        self.cluster_assignments = torch.arange(
            self.num_tiles, device=signatures.device
        ) // self.tiles_per_cluster
        
        # Compute cluster signatures (mean of member signatures)
        cluster_sigs = []
        for c in range(self.num_clusters):
            mask = self.cluster_assignments == c
            cluster_sigs.append(signatures[mask].mean(dim=0).sign())
        self.cluster_signatures = torch.stack(cluster_sigs)
        
        self._hierarchy_built = True
    
    def _route_hierarchical(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Two-level hierarchical routing.
        
        Returns:
            tile_indices: [batch, seq] - which tile each token routes to
            routing_weights: [batch, seq] - confidence weights
            stats: routing statistics
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Build/update hierarchy
        if not self._hierarchy_built or self.training:
            self._build_hierarchy()
        
        # Compute input signatures
        x_flat = x.view(-1, d_model)
        input_sigs = x_flat.sign()  # [batch*seq, d_model]
        
        # Level 1: Route to cluster
        cluster_scores = torch.matmul(
            input_sigs.float(),
            self.cluster_signatures.t().float()
        )  # [batch*seq, num_clusters]
        
        best_clusters = cluster_scores.argmax(dim=-1)  # [batch*seq]
        
        # Level 2: Route to tile within cluster
        tile_indices = torch.zeros(batch_size * seq_len, dtype=torch.long, device=device)
        routing_weights = torch.zeros(batch_size * seq_len, device=device)
        
        for c in range(self.num_clusters):
            cluster_mask = best_clusters == c
            if not cluster_mask.any():
                continue
            
            # Get tiles in this cluster
            tile_mask = self.cluster_assignments == c
            cluster_tile_indices = torch.where(tile_mask)[0]
            cluster_tile_sigs = self.ema_tile_signatures[tile_mask].sign()
            
            # Route within cluster
            cluster_inputs = input_sigs[cluster_mask]
            tile_scores = torch.matmul(
                cluster_inputs.float(),
                cluster_tile_sigs.t().float()
            )
            
            local_best = tile_scores.argmax(dim=-1)
            tile_indices[cluster_mask] = cluster_tile_indices[local_best]
            routing_weights[cluster_mask] = tile_scores.max(dim=-1).values
        
        # Normalize weights
        routing_weights = torch.softmax(routing_weights.view(batch_size, seq_len), dim=-1)
        tile_indices = tile_indices.view(batch_size, seq_len)
        
        # Stats
        stats = {
            'unique_tiles': tile_indices.unique().numel(),
            'cluster_balance': best_clusters.view(batch_size, seq_len).float().std().item(),
        }
        
        return tile_indices, routing_weights.view(batch_size, seq_len), stats
    
    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with hierarchical routing.
        
        Args:
            x: [batch, seq, d_model]
            return_stats: whether to return routing statistics
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Normalize for routing
        x_norm = self.input_norm(x)
        
        # Route
        tile_indices, routing_weights, stats = self._route_hierarchical(x_norm)
        
        # Compute outputs
        output = torch.zeros_like(x)
        
        for tile_idx in range(self.num_tiles):
            mask = tile_indices == tile_idx
            if not mask.any():
                continue
            
            # Get inputs for this tile
            tile_input = x_norm[mask]  # [num_routed, d_model]
            
            # Compute through KAN tile
            tile_output = self.tiles[tile_idx](tile_input)
            
            # Place outputs
            output[mask] = tile_output
            
            # Update usage stats
            self.tiles[tile_idx].update_usage(mask.sum().item(), batch_size * seq_len)
        
        output = self.dropout(output)
        
        # Global residual (in addition to per-tile residual)
        if self.use_residual:
            output = x + output
        
        if return_stats:
            return output, stats
        return output
    
    def get_routing_stats(self) -> Dict:
        """Get comprehensive routing statistics."""
        usage_rates = [tile.usage_rate for tile in self.tiles]
        
        return {
            'num_tiles': self.num_tiles,
            'num_clusters': self.num_clusters,
            'grid_size': self.grid_size,
            'usage_rates': usage_rates,
            'usage_mean': sum(usage_rates) / len(usage_rates),
            'usage_std': torch.tensor(usage_rates).std().item(),
            'active_tiles': sum(1 for r in usage_rates if r > 0.01),
        }
    
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def memory_report(self) -> Dict:
        """Memory usage report."""
        total_params = self.total_parameters()
        tile_params = sum(tile.num_parameters() for tile in self.tiles)
        
        return {
            'total_params': total_params,
            'tile_params': tile_params,
            'memory_fp32': total_params * 4,
            'memory_fp16': total_params * 2,
            'memory_2bit': total_params // 4,
            'per_tile_params': tile_params // self.num_tiles,
            'per_tile_2bit': (tile_params // self.num_tiles) // 4,
        }


def test_kan_hierarchical():
    """Test HierarchicalKANFFN."""
    print("=" * 60)
    print("HIERARCHICAL KAN FFN TEST")
    print("=" * 60)
    
    # Create model
    d_model = 256
    num_tiles = 64
    grid_size = 16
    
    model = HierarchicalKANFFN(
        d_model=d_model,
        num_tiles=num_tiles,
        tiles_per_cluster=8,
        grid_size=grid_size,
    )
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_tiles: {num_tiles}")
    print(f"  grid_size: {grid_size}")
    
    # Memory report
    mem = model.memory_report()
    print(f"\nMemory:")
    print(f"  Total params: {mem['total_params']:,}")
    print(f"  Tile params: {mem['tile_params']:,}")
    print(f"  Per tile (2-bit): {mem['per_tile_2bit']:,} bytes")
    print(f"  Total (2-bit): {mem['memory_2bit']:,} bytes")
    
    # Forward pass
    print(f"\nForward pass:")
    batch_size, seq_len = 4, 32
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, stats = model(x, return_stats=True)
    
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Unique tiles used: {stats['unique_tiles']}")
    
    # Routing stats
    routing = model.get_routing_stats()
    print(f"\nRouting:")
    print(f"  Active tiles: {routing['active_tiles']} / {routing['num_tiles']}")
    print(f"  Usage mean: {routing['usage_mean']:.3f}")
    print(f"  Usage std: {routing['usage_std']:.3f}")
    
    # Gradient check
    print(f"\nGradient check:")
    loss = output.sum()
    loss.backward()
    
    grad_norms = []
    for tile in model.tiles:
        for p in tile.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
    
    print(f"  Grad norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}")
    
    # Compare to MLP baseline
    print(f"\nComparison to MLP:")
    mlp_params = d_model * (d_model * 4) * 2 + d_model * 4 + d_model  # Standard FFN
    print(f"  MLP FFN params: {mlp_params:,}")
    print(f"  KAN FFN params: {mem['total_params']:,}")
    print(f"  Savings: {(1 - mem['total_params'] / mlp_params) * 100:.1f}%")
    
    return True


if __name__ == '__main__':
    test_kan_hierarchical()
