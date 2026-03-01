"""
Hybrid KAN: The Best of Both Worlds

Combines:
- Hierarchical routing (O(√n) addressing)
- Hybrid tiles (bottleneck + spline)
- VGem's Residual Bus

This is the PQH architecture that LEARNS.

Architecture per tile:
  Input (d_model) → Down (d_model/4) → Spline → Up (d_model) → Output
  
With 64 tiles and routing, this gives:
- Sparse computation (only active tile computes)
- Learnable nonlinearity (splines adapt)
- Efficient parameters (bottleneck reduces size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class HybridKANTile(nn.Module):
    """
    Hybrid KAN tile: bottleneck + spline + upproject.
    
    This is expressive enough to learn while being sparse.
    """
    
    def __init__(
        self,
        d_model: int,
        bottleneck_ratio: int = 4,
        grid_size: int = 16,
        tile_id: int = 0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_hidden = d_model // bottleneck_ratio
        self.grid_size = grid_size
        self.tile_id = tile_id
        
        # Bottleneck projection
        self.down = nn.Linear(d_model, self.d_hidden, bias=False)
        
        # Spline parameters for each hidden dimension
        self.spline_bases = nn.Parameter(torch.zeros(self.d_hidden, grid_size))
        self.spline_slopes = nn.Parameter(torch.randn(self.d_hidden, grid_size) * 0.1)
        
        # Up projection
        self.up = nn.Linear(self.d_hidden, d_model, bias=False)
        
        # Output scale (VGem's gradient knob)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Usage tracking
        self.register_buffer('activation_count', torch.tensor(0.0))
        self.register_buffer('total_count', torch.tensor(0.0))
        
    def get_signature(self) -> torch.Tensor:
        """Signature from down projection weights."""
        with torch.no_grad():
            return self.down.weight.mean(dim=0).sign()
    
    def _spline_transform(self, h: torch.Tensor) -> torch.Tensor:
        """Apply learnable spline nonlinearity."""
        # Normalize to [0, 1) for grid indexing
        h_norm = torch.sigmoid(h)
        
        # Grid indices
        idx = (h_norm * self.grid_size).long().clamp(0, self.grid_size - 1)
        
        # Gather coefficients
        batch_dims = h.shape[:-1]
        
        # Reshape for gathering
        bases = self.spline_bases.view(1, self.d_hidden, self.grid_size)
        slopes = self.spline_slopes.view(1, self.d_hidden, self.grid_size)
        
        # Expand to batch
        bases = bases.expand(*batch_dims, -1, -1)
        slopes = slopes.expand(*batch_dims, -1, -1)
        
        idx_expanded = idx.unsqueeze(-1)
        
        base = bases.gather(-1, idx_expanded).squeeze(-1)
        slope = slopes.gather(-1, idx_expanded).squeeze(-1)
        
        # Local position within cell
        cell_size = 1.0 / self.grid_size
        local = (h_norm - idx.float() * cell_size) / cell_size
        
        return base + slope * local
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: down → spline → up."""
        h = self.down(x)
        h = self._spline_transform(h)
        out = self.up(h) * self.scale
        return out
    
    def update_usage(self, count: int, total: int):
        self.activation_count += count
        self.total_count += total
    
    @property
    def usage_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.activation_count / self.total_count).item()
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HybridKANFFN(nn.Module):
    """
    Hierarchical Hybrid KAN FFN.
    
    2-level routing + hybrid KAN tiles.
    """
    
    def __init__(
        self,
        d_model: int,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        bottleneck_ratio: int = 4,
        grid_size: int = 16,
        dropout: float = 0.1,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.tiles_per_cluster = tiles_per_cluster
        self.num_clusters = num_tiles // tiles_per_cluster
        self.ema_decay = ema_decay
        
        # Input normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Hybrid tiles
        self.tiles = nn.ModuleList([
            HybridKANTile(d_model, bottleneck_ratio, grid_size, tile_id=i)
            for i in range(num_tiles)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # EMA signatures
        self.register_buffer('ema_signatures', None)
        self.register_buffer('cluster_signatures', None)
        self.register_buffer('cluster_assignments', None)
        
    def _update_signatures(self):
        """Update EMA signatures from tiles."""
        current = torch.stack([t.get_signature() for t in self.tiles])
        
        if self.ema_signatures is None:
            self.ema_signatures = current.clone()
        else:
            self.ema_signatures = self.ema_decay * self.ema_signatures + (1 - self.ema_decay) * current
        
        # Build clusters (simple: consecutive grouping)
        self.cluster_assignments = torch.arange(self.num_tiles, device=current.device) // self.tiles_per_cluster
        
        # Cluster signatures
        cluster_sigs = []
        for c in range(self.num_clusters):
            mask = self.cluster_assignments == c
            cluster_sigs.append(self.ema_signatures[mask].mean(dim=0).sign())
        self.cluster_signatures = torch.stack(cluster_sigs)
    
    def _route(self, x: torch.Tensor) -> torch.Tensor:
        """Hierarchical routing: cluster → tile."""
        # x: [batch, seq, d_model]
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        self._update_signatures()
        
        x_flat = x.view(-1, d_model)
        input_sigs = x_flat.sign()
        
        # Level 1: Cluster
        cluster_scores = input_sigs.float() @ self.cluster_signatures.t().float()
        best_clusters = cluster_scores.argmax(dim=-1)
        
        # Level 2: Tile within cluster
        tile_indices = torch.zeros(batch_size * seq_len, dtype=torch.long, device=device)
        
        for c in range(self.num_clusters):
            mask = best_clusters == c
            if not mask.any():
                continue
            
            tile_mask = self.cluster_assignments == c
            cluster_tiles = torch.where(tile_mask)[0]
            cluster_sigs = self.ema_signatures[tile_mask].sign()
            
            scores = input_sigs[mask].float() @ cluster_sigs.t().float()
            local_best = scores.argmax(dim=-1)
            tile_indices[mask] = cluster_tiles[local_best]
        
        return tile_indices.view(batch_size, seq_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with routing."""
        batch_size, seq_len, d_model = x.shape
        
        x_norm = self.norm(x)
        tile_indices = self._route(x_norm)
        
        # Compute through routed tiles
        output = torch.zeros_like(x)
        
        for tile_idx in range(self.num_tiles):
            mask = tile_indices == tile_idx
            if not mask.any():
                continue
            
            tile_input = x_norm[mask]
            tile_output = self.tiles[tile_idx](tile_input)
            output[mask] = tile_output
            
            self.tiles[tile_idx].update_usage(mask.sum().item(), batch_size * seq_len)
        
        output = self.dropout(output)
        
        # Residual
        return x + output
    
    def get_stats(self) -> Dict:
        usage = [t.usage_rate for t in self.tiles]
        return {
            'num_tiles': self.num_tiles,
            'active_tiles': sum(1 for u in usage if u > 0.01),
            'usage_mean': sum(usage) / len(usage),
            'usage_std': torch.tensor(usage).std().item(),
        }
    
    def memory_report(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        tile_params = sum(t.num_parameters() for t in self.tiles)
        
        return {
            'total_params': total,
            'tile_params': tile_params,
            'per_tile': tile_params // self.num_tiles,
            'bytes_fp32': total * 4,
            'bytes_2bit': total // 4,
        }


def test_hybrid_kan():
    """Test HybridKANFFN."""
    print("=" * 60)
    print("HYBRID KAN FFN TEST")
    print("=" * 60)
    
    d_model = 256
    model = HybridKANFFN(
        d_model=d_model,
        num_tiles=64,
        tiles_per_cluster=8,
        bottleneck_ratio=4,
        grid_size=16,
    )
    
    mem = model.memory_report()
    print(f"\nMemory:")
    print(f"  Total params: {mem['total_params']:,}")
    print(f"  Per tile: {mem['per_tile']:,}")
    print(f"  2-bit total: {mem['bytes_2bit']:,} bytes")
    
    # Forward
    x = torch.randn(4, 32, d_model)
    out = model(x)
    
    print(f"\nForward:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    
    # Gradient
    loss = out.sum()
    loss.backward()
    
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  Gradients: {'OK' if grad_ok else 'FAIL'}")
    
    # Compare to MLP
    mlp_params = d_model * d_model * 4 * 2 + d_model * 4 + d_model
    print(f"\nComparison:")
    print(f"  MLP FFN: {mlp_params:,} params")
    print(f"  Hybrid KAN: {mem['total_params']:,} params")
    print(f"  Savings: {(1 - mem['total_params']/mlp_params)*100:.1f}%")
    
    return True


if __name__ == '__main__':
    test_hybrid_kan()
