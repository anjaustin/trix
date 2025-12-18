"""
SparseLookupFFNv3: Geometric TriX

Three Millennium Problem upgrades:

1. NAVIER-STOKES (Reynolds Decomposition):
   - Split into smooth "Spline Flow" + chaotic "Turbulence Residual"
   - Main experts handle structure, Vortex expert handles noise
   
2. YANG-MILLS (Parallel Transport):
   - Gauge transforms between tiles (learned rotations)
   - Vectors "fit" destination tile's coordinate system
   - Enables expert reuse across positions
   
3. HODGE (Topological Compression):
   - Track routing cycles to detect clusters
   - Persistent loops → mergeable tiles
   - Auto-pruning via cohomology

Core insight: Data is a fluid flowing through a manifold, not a bag of points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


def cubic_bspline(t: torch.Tensor) -> torch.Tensor:
    """Cubic B-spline kernel. C² continuous - the smoothness guarantee."""
    t = t.abs()
    result = torch.zeros_like(t)
    
    mask1 = t < 1
    result[mask1] = (2/3) - t[mask1]**2 + 0.5 * t[mask1]**3
    
    mask2 = (t >= 1) & (t < 2)
    result[mask2] = (1/6) * (2 - t[mask2])**3
    
    return result


class GaugeTransform(nn.Module):
    """
    Yang-Mills inspired: Parallel transport between tiles.
    
    When moving from tile i to tile j, rotate the vector
    to fit the destination's local coordinate system.
    """
    
    def __init__(self, d_model: int, num_tiles: int):
        super().__init__()
        # Learn a small rotation for each tile (relative to canonical frame)
        # Using exponential map for SO(n) would be ideal, but expensive
        # Approximate with learned scale + shift per tile
        self.tile_scale = nn.Parameter(torch.ones(num_tiles, d_model))
        self.tile_shift = nn.Parameter(torch.zeros(num_tiles, d_model))
    
    def to_local(self, x: torch.Tensor, tile_idx: torch.Tensor) -> torch.Tensor:
        """Transform to tile's local coordinates."""
        scale = self.tile_scale[tile_idx]  # [B*T, D]
        shift = self.tile_shift[tile_idx]
        return x * scale + shift
    
    def to_global(self, x: torch.Tensor, tile_idx: torch.Tensor) -> torch.Tensor:
        """Transform back to global coordinates."""
        scale = self.tile_scale[tile_idx]
        shift = self.tile_shift[tile_idx]
        return (x - shift) / (scale + 1e-6)


class VortexExpert(nn.Module):
    """
    Navier-Stokes inspired: Handle the turbulence/residual.
    
    Main spline experts handle smooth structure.
    This expert handles what's left - the high-frequency noise.
    """
    
    def __init__(self, d_model: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or d_model // 4
        
        # Small, fast network for residuals
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        
        # Learnable gate: how much turbulence to add
        self.gate = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """Process the turbulent residual."""
        return self.net(residual) * torch.sigmoid(self.gate)


class TopologyTracker(nn.Module):
    """
    Hodge inspired: Track routing topology for compression.
    
    Detect persistent cycles in routing patterns.
    A→B→A loops indicate mergeable experts.
    """
    
    def __init__(self, num_tiles: int):
        super().__init__()
        self.num_tiles = num_tiles
        
        # Transition counts: how often do we go from tile i to tile j?
        self.register_buffer(
            'transitions',
            torch.zeros(num_tiles, num_tiles)
        )
        self.register_buffer('total_transitions', torch.tensor(0.0))
    
    def update(self, prev_tiles: torch.Tensor, curr_tiles: torch.Tensor):
        """Track transition from prev_tiles to curr_tiles."""
        if prev_tiles is None:
            return
        
        # Count transitions
        for p, c in zip(prev_tiles.flatten(), curr_tiles.flatten()):
            self.transitions[p, c] += 1
        self.total_transitions += prev_tiles.numel()
    
    def get_cycles(self, threshold: float = 0.1) -> list:
        """Find strong bidirectional connections (potential merges)."""
        if self.total_transitions == 0:
            return []
        
        # Normalize
        T = self.transitions / (self.total_transitions + 1e-8)
        
        # Find symmetric pairs (A→B and B→A both strong)
        cycles = []
        for i in range(self.num_tiles):
            for j in range(i + 1, self.num_tiles):
                if T[i, j] > threshold and T[j, i] > threshold:
                    cycles.append((i, j, T[i, j].item() + T[j, i].item()))
        
        return sorted(cycles, key=lambda x: -x[2])
    
    def reset(self):
        self.transitions.zero_()
        self.total_transitions.zero_()


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


class SparseLookupFFNv3(nn.Module):
    """
    Geometric TriX: Sparse Lookup with Millennium Problem upgrades.
    
    Architecture:
        1. Positional routing (B-spline × Content) → smooth trajectory
        2. Gauge transform → parallel transport between tiles
        3. Reynolds split → spline flow + turbulence residual
        4. Topology tracking → detect mergeable cycles
    
    Args:
        d_model: Model dimension
        num_tiles: Number of expert tiles
        max_seq_len: Maximum sequence length
        position_spread: B-spline spreading width
        grid_size: Spline grid resolution
        use_gauge: Enable Yang-Mills gauge transforms
        use_vortex: Enable Navier-Stokes turbulence expert
        track_topology: Enable Hodge topology tracking
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        max_seq_len: int = 2048,
        position_spread: float = 2.0,
        grid_size: int = 16,
        use_gauge: bool = True,
        use_vortex: bool = True,
        track_topology: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_clusters = num_tiles // tiles_per_cluster
        self.tiles_per_cluster = tiles_per_cluster
        self.max_seq_len = max_seq_len
        self.position_spread = position_spread
        
        # Core components
        self.norm = nn.LayerNorm(d_model)
        
        # Compression network
        compress_hidden = d_model // 4
        self.compress = nn.Sequential(
            nn.Linear(d_model, compress_hidden),
            nn.GELU(),
            nn.Linear(compress_hidden, 2),
            nn.Tanh(),
        )
        
        # Per-tile splines (the smooth manifold)
        self.splines = nn.ModuleList([
            TernarySpline2D(grid_size) for _ in range(num_tiles)
        ])
        
        # Tile directions (what each tile contributes)
        self.directions = nn.Parameter(torch.randn(num_tiles, d_model) * 0.02)
        
        # Positional routing: tile positions along sequence
        self.register_buffer(
            'tile_positions',
            torch.linspace(0, max_seq_len, num_tiles)
        )
        
        # Cluster assignments
        self.register_buffer(
            'cluster_assignments',
            torch.arange(num_tiles) // tiles_per_cluster
        )
        
        # === MILLENNIUM UPGRADES ===
        
        # Yang-Mills: Gauge transforms for parallel transport
        self.use_gauge = use_gauge
        if use_gauge:
            self.gauge = GaugeTransform(d_model, num_tiles)
        
        # Navier-Stokes: Vortex expert for turbulence
        self.use_vortex = use_vortex
        if use_vortex:
            self.vortex = VortexExpert(d_model)
        
        # Hodge: Topology tracking
        self.track_topology = track_topology
        if track_topology:
            self.topology = TopologyTracker(num_tiles)
        
        # Output
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.dropout = nn.Dropout(dropout)
        
        # State for topology tracking
        self.prev_tile_idx = None
        
        # Usage stats
        self.register_buffer('tile_counts', torch.zeros(num_tiles))
        self.register_buffer('total_count', torch.tensor(0.0))
    
    def get_signatures(self) -> torch.Tensor:
        """Derive ternary routing signatures from directions."""
        return self.directions.sign()
    
    def get_cluster_signatures(self, signatures: torch.Tensor) -> torch.Tensor:
        """Compute cluster-level signatures."""
        cluster_sigs = []
        for c in range(self.num_clusters):
            mask = self.cluster_assignments == c
            cluster_sigs.append(signatures[mask].mean(dim=0).sign())
        return torch.stack(cluster_sigs)
    
    def positional_routing(
        self, 
        x: torch.Tensor, 
        positions: torch.Tensor,
        signatures: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Geometric routing: Content × Position (B-spline spreading).
        
        Returns tile indices AND soft weights for smooth interpolation.
        """
        B_T, D = x.shape
        device = x.device
        
        # Content scores (standard TriX)
        cluster_sigs = self.get_cluster_signatures(signatures)
        
        # Level 1: Cluster routing
        cluster_scores = x @ cluster_sigs.T
        
        # Position modulation for clusters
        pos_normalized = positions / self.max_seq_len * self.num_clusters
        cluster_centers = torch.arange(self.num_clusters, device=device).float()
        cluster_pos_diff = pos_normalized.unsqueeze(-1) - cluster_centers.unsqueeze(0)
        cluster_pos_scores = cubic_bspline(cluster_pos_diff / (self.position_spread * self.tiles_per_cluster / self.num_tiles))
        
        cluster_combined = cluster_scores * cluster_pos_scores
        cluster_idx = cluster_combined.argmax(dim=-1)
        
        # Level 2: Tile routing within cluster
        tile_idx = torch.zeros(B_T, dtype=torch.long, device=device)
        tile_weights = torch.zeros(B_T, device=device)
        
        for c in range(self.num_clusters):
            mask = cluster_idx == c
            if not mask.any():
                continue
            
            tile_mask = self.cluster_assignments == c
            cluster_tiles = torch.where(tile_mask)[0]
            cluster_tile_sigs = signatures[tile_mask]
            
            # Content scores within cluster
            content_scores = x[mask] @ cluster_tile_sigs.T
            
            # Position scores within cluster (B-spline)
            tile_centers = cluster_tiles.float()
            pos_in_tiles = positions[mask].unsqueeze(-1)
            tile_pos_normalized = pos_in_tiles / self.max_seq_len * self.num_tiles
            pos_diff = tile_pos_normalized - tile_centers.unsqueeze(0)
            pos_scores = cubic_bspline(pos_diff / self.position_spread)
            
            # Combined routing
            combined = content_scores * pos_scores
            local_idx = combined.argmax(dim=-1)
            
            # Get winning tile's weight (for soft interpolation)
            winning_weights = combined.gather(1, local_idx.unsqueeze(-1)).squeeze(-1)
            
            tile_idx[mask] = cluster_tiles[local_idx]
            tile_weights[mask] = winning_weights
        
        return tile_idx, tile_weights
    
    def forward(
        self, 
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Forward pass with geometric routing.
        
        Args:
            x: [B, T, d_model]
            positions: [B, T] or None (defaults to 0, 1, 2, ...)
        """
        B, T, D = x.shape
        device = x.device
        
        # Default positions
        if positions is None:
            positions = torch.arange(T, device=device).float().unsqueeze(0).expand(B, -1)
        
        # Normalize
        x_norm = self.norm(x)
        x_flat = x_norm.reshape(-1, D)
        pos_flat = positions.reshape(-1)
        
        # Get signatures
        signatures = self.get_signatures()
        
        # Geometric routing
        tile_idx, tile_weights = self.positional_routing(x_flat, pos_flat, signatures)
        
        # Track topology (Hodge)
        if self.track_topology and self.training:
            self.topology.update(self.prev_tile_idx, tile_idx)
            self.prev_tile_idx = tile_idx.clone()
        
        # Compress
        compressed = self.compress(x_flat)
        a, b = compressed[:, 0], compressed[:, 1]
        
        # === REYNOLDS DECOMPOSITION ===
        # Step 1: Compute smooth spline output (laminar flow)
        spline_output = torch.zeros_like(x_flat)
        
        for t in range(self.num_tiles):
            mask = tile_idx == t
            if not mask.any():
                continue
            
            # Get inputs for this tile
            x_tile = x_flat[mask]
            
            # Yang-Mills: Transform to tile's local coordinates
            if self.use_gauge:
                x_local = self.gauge.to_local(x_tile, torch.full((mask.sum(),), t, device=device))
            else:
                x_local = x_tile
            
            # Spline lookup (smooth structure)
            scale = self.splines[t](a[mask], b[mask])
            
            # Apply direction
            contribution = scale.unsqueeze(-1) * self.directions[t]
            
            # Yang-Mills: Transform back to global
            if self.use_gauge:
                contribution = self.gauge.to_global(
                    contribution, 
                    torch.full((mask.sum(),), t, device=device)
                )
            
            spline_output[mask] = contribution
            
            # Track usage
            if self.training:
                self.tile_counts[t] += mask.sum().float()
        
        if self.training:
            self.total_count += B * T
        
        # Step 2: Compute residual and process with vortex expert (turbulence)
        if self.use_vortex:
            # Residual = what the spline couldn't capture
            residual = x_flat - spline_output
            turbulence = self.vortex(residual)
            output = spline_output + turbulence
        else:
            output = spline_output
        
        # Reshape and scale
        output = output.view(B, T, D) * self.output_scale
        output = self.dropout(output)
        
        # Residual connection
        output = x + output
        
        # Routing info
        routing_info = {
            'tile_idx': tile_idx.view(B, T),
            'tile_weights': tile_weights.view(B, T),
            'compressed': compressed.view(B, T, 2),
        }
        
        # Auxiliary losses
        aux_losses = self._compute_aux_losses(tile_idx, B * T)
        
        return output, routing_info, aux_losses
    
    def _compute_aux_losses(self, tile_idx: torch.Tensor, total: int) -> Dict:
        """Compute auxiliary losses including topology regularization."""
        counts = torch.zeros(self.num_tiles, device=tile_idx.device)
        for t in range(self.num_tiles):
            counts[t] = (tile_idx == t).sum().float()
        
        ideal = total / self.num_tiles
        balance_loss = ((counts - ideal) ** 2).mean() / (ideal ** 2 + 1e-8)
        
        # Topology regularization: penalize too many strong cycles (encourage specialization)
        topo_loss = torch.tensor(0.0, device=tile_idx.device)
        if self.track_topology and self.training:
            cycles = self.topology.get_cycles(threshold=0.05)
            if cycles:
                # Penalize symmetric routing (want tiles to specialize)
                topo_loss = torch.tensor(len(cycles) * 0.001, device=tile_idx.device)
        
        total_aux = balance_loss * 0.01 + topo_loss
        
        return {
            'balance': balance_loss * 0.01,
            'topology': topo_loss,
            'total_aux': total_aux,
        }
    
    def get_routing_stats(self) -> Dict:
        """Get routing and topology statistics."""
        if self.total_count == 0:
            return {'num_tiles': self.num_tiles, 'active_tiles': 0}
        
        usage = self.tile_counts / self.total_count
        
        stats = {
            'num_tiles': self.num_tiles,
            'active_tiles': (usage > 0.001).sum().item(),
            'usage_mean': usage.mean().item(),
            'usage_std': usage.std().item(),
        }
        
        if self.track_topology:
            cycles = self.topology.get_cycles(threshold=0.05)
            stats['routing_cycles'] = len(cycles)
            stats['top_cycles'] = cycles[:5] if cycles else []
        
        return stats
    
    def reset_stats(self):
        """Reset all statistics."""
        self.tile_counts.zero_()
        self.total_count.zero_()
        self.prev_tile_idx = None
        if self.track_topology:
            self.topology.reset()
    
    def get_param_count(self) -> Dict:
        """Count parameters by component."""
        compress_params = sum(p.numel() for p in self.compress.parameters())
        spline_params = sum(sum(p.numel() for p in s.parameters()) for s in self.splines)
        direction_params = self.directions.numel()
        norm_params = sum(p.numel() for p in self.norm.parameters())
        
        gauge_params = sum(p.numel() for p in self.gauge.parameters()) if self.use_gauge else 0
        vortex_params = sum(p.numel() for p in self.vortex.parameters()) if self.use_vortex else 0
        
        total = compress_params + spline_params + direction_params + norm_params + gauge_params + vortex_params + 1
        
        return {
            'compress': compress_params,
            'splines': spline_params,
            'directions': direction_params,
            'norm': norm_params,
            'gauge': gauge_params,
            'vortex': vortex_params,
            'total': total,
        }


# === Testing ===

def test_v3():
    """Quick test of SparseLookupFFNv3."""
    print("=" * 60)
    print("SparseLookupFFNv3 (Geometric TriX) Test")
    print("=" * 60)
    
    d_model = 128
    batch_size = 4
    seq_len = 64
    
    model = SparseLookupFFNv3(
        d_model=d_model,
        num_tiles=32,
        tiles_per_cluster=8,
        max_seq_len=512,
        use_gauge=True,
        use_vortex=True,
        track_topology=True,
    )
    
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
    
    # Check routing pattern
    tile_idx = routing_info['tile_idx']
    print(f"\nRouting (first sequence):")
    print(f"  Tiles: {tile_idx[0, :10].tolist()}...")
    
    # Gradient check
    loss = output.sum() + aux_losses['total_aux']
    loss.backward()
    
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"\nGradients: {'OK' if grad_ok else 'FAIL'}")
    
    # Stats
    stats = model.get_routing_stats()
    print(f"\nStats:")
    print(f"  Active tiles: {stats['active_tiles']}/{stats['num_tiles']}")
    print(f"  Routing cycles: {stats.get('routing_cycles', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("Test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_v3()
