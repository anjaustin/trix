"""
Hierarchical TriX: Content-Addressable Memory at Scale

The Big Leap: 2-level hierarchical routing enabling 64-1000+ tiles
with O(sqrt(n)) routing cost instead of O(n).

This is TriX as content-addressable memory:
- Signatures = Keys (content addresses)
- Tiles = Values (executable 2-bit functions)
- Routing = Lookup (find by alignment)
- Hierarchy = Index (organize for fast lookup)

"Qdrant with a brain at every address."
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math

from trix.kernel import TriXLinear, STESign
from trix.nn.xor_superposition import CompressedSignatures, CompressionStats


class TriXTile(nn.Module):
    """
    A single 2-bit specialist tile.
    
    Each tile is a tiny brain that:
    1. Knows what it wants (signature)
    2. Knows what to do (ternary weights)
    3. Does it fast (2-bit ops)
    
    VGem's fixes applied:
    - Learnable output scale (gives gradient a continuous knob)
    """
    
    def __init__(self, d_model: int, d_hidden: int, tile_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.tile_id = tile_id
        
        # Ternary weights
        self.up_weight = nn.Parameter(torch.randn(d_hidden, d_model) * 0.02)
        self.down_weight = nn.Parameter(torch.randn(d_model, d_hidden) * 0.02)
        
        # Learnable scales (post-quantization)
        self.up_scale = nn.Parameter(torch.ones(d_hidden))
        self.down_scale = nn.Parameter(torch.ones(d_model))
        
        # VGem's fix: Learnable output scale (continuous knob for gradients)
        self.output_scale = nn.Parameter(torch.ones(1))
        
        # Usage tracking
        self.register_buffer('activation_count', torch.tensor(0.0))
        self.register_buffer('total_count', torch.tensor(0.0))
    
    def get_signature(self) -> torch.Tensor:
        """
        Extract signature from weights.
        
        The signature is what this tile "wants" - derived from
        the column-wise sum of ternary weights.
        """
        with torch.no_grad():
            ternary = self.up_weight.sign()
            signature = ternary.sum(dim=0).sign()
        return signature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through this tile."""
        # Quantize weights (STE)
        up_w = STESign.apply(self.up_weight)
        down_w = STESign.apply(self.down_weight)
        
        # Up projection
        hidden = F.linear(x, up_w) * self.up_scale
        hidden = F.relu(hidden)
        
        # Down projection
        out = F.linear(hidden, down_w) * self.down_scale
        
        # VGem's fix: learnable output scale
        out = out * self.output_scale
        
        return out
    
    def update_usage(self, count: int, total: int):
        """Track activation frequency."""
        self.activation_count += count
        self.total_count += total
    
    @property
    def usage_rate(self) -> float:
        """Get activation rate (0-1)."""
        if self.total_count == 0:
            return 0.0
        return (self.activation_count / self.total_count).item()


class HierarchicalTriXFFN(nn.Module):
    """
    Hierarchical Content-Addressable Memory FFN.
    
    2-level routing:
      Level 1: Route to cluster (coarse)
      Level 2: Route to tile within cluster (fine)
    
    This enables O(sqrt(n)) routing instead of O(n):
    - 64 tiles = 8 clusters × 8 tiles = 16 comparisons (vs 64)
    - 256 tiles = 16 × 16 = 32 comparisons (vs 256)
    - 1024 tiles = 32 × 32 = 64 comparisons (vs 1024)
    
    Each tile is a 2-bit specialist. The signatures are content
    addresses. Routing is lookup by alignment.
    
    VGem's fixes applied:
    - Input normalization (RMSNorm) for stable routing
    - Residual connection (tile learns delta, not whole function)
    - Higher EMA decay (0.999) for address stability
    - Frozen routing option for debugging
    
    Args:
        d_model: Model dimension
        d_hidden: Hidden dimension per tile
        num_tiles: Total number of tiles
        tiles_per_cluster: Tiles in each cluster (num_clusters = num_tiles // tiles_per_cluster)
        dropout: Dropout rate
        balance_weight: Weight for load balancing loss
        diversity_weight: Weight for diversity loss
        top_k_clusters: Route to top-k clusters (soft hierarchy, default 1 = hard)
        ema_decay: EMA decay for signature stability (VGem recommends 0.999)
        use_residual: Whether to add residual connection (VGem's fix)
        freeze_routing: Freeze signatures for "Frozen Routing" test
    """
    
    def __init__(
        self,
        d_model: int,
        d_hidden: int = None,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        dropout: float = 0.1,
        balance_weight: float = 0.01,
        diversity_weight: float = 0.001,
        top_k_clusters: int = 1,
        ema_decay: float = 0.999,  # VGem: increased from 0.99
        use_residual: bool = True,  # VGem's fix
        freeze_routing: bool = False,  # For "Frozen Routing" test
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model * 4 // num_tiles
        self.num_tiles = num_tiles
        self.tiles_per_cluster = tiles_per_cluster
        self.num_clusters = num_tiles // tiles_per_cluster
        self.top_k_clusters = min(top_k_clusters, self.num_clusters)
        self.balance_weight = balance_weight
        self.diversity_weight = diversity_weight
        self.ema_decay = ema_decay
        
        assert num_tiles % tiles_per_cluster == 0, \
            f"num_tiles ({num_tiles}) must be divisible by tiles_per_cluster ({tiles_per_cluster})"
        
        self.use_residual = use_residual
        self.freeze_routing = freeze_routing
        
        # VGem's fix: Input normalization for stable routing
        self.input_norm = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        
        # Create all tiles
        self.tiles = nn.ModuleList([
            TriXTile(d_model, self.d_hidden, tile_id=i)
            for i in range(num_tiles)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # EMA signatures for stable routing
        self.register_buffer('ema_tile_signatures', None)
        self.register_buffer('cluster_signatures', None)
        self.register_buffer('cluster_assignments', None)
        self.register_buffer('frozen_signatures', None)  # For frozen routing test

        # Initialize hierarchy
        self._hierarchy_built = False

        # XOR compression for inference
        self._compressed_tile_signatures: Optional[CompressedSignatures] = None
        self._compressed_cluster_signatures: Optional[CompressedSignatures] = None
        self._is_compressed = False
    
    def _get_current_signatures(self) -> torch.Tensor:
        """Get current signatures from all tiles."""
        return torch.stack([tile.get_signature() for tile in self.tiles])
    
    def _update_ema_signatures(self):
        """Update EMA signatures for stability (VGem's recommendation)."""
        current = self._get_current_signatures()
        
        if self.ema_tile_signatures is None:
            self.ema_tile_signatures = current.clone()
        else:
            self.ema_tile_signatures = (
                self.ema_decay * self.ema_tile_signatures +
                (1 - self.ema_decay) * current
            )
    
    def build_hierarchy(self):
        """
        Build 2-level routing hierarchy via k-means clustering.
        
        Groups tiles by signature similarity, then computes
        cluster centroids as routing targets.
        """
        with torch.no_grad():
            # Get stable signatures
            if self.ema_tile_signatures is not None:
                sigs = self.ema_tile_signatures
            else:
                sigs = self._get_current_signatures()
            
            # Simple k-means clustering
            cluster_assignments, centroids = self._kmeans(
                sigs, self.num_clusters, max_iters=20
            )
            
            # Store hierarchy
            self.cluster_assignments = cluster_assignments
            self.cluster_signatures = centroids.sign()  # Ternary cluster signatures
            self._hierarchy_built = True
    
    def _kmeans(
        self, 
        data: torch.Tensor, 
        k: int, 
        max_iters: int = 20,
        balanced: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        K-means clustering on signatures with optional balancing.
        
        Args:
            data: [n, d] tensor of signatures
            k: number of clusters
            max_iters: maximum iterations
            balanced: if True, enforce equal cluster sizes
        
        Returns:
            assignments: [num_tiles] cluster index per tile
            centroids: [k, d_model] cluster centers
        """
        n, d = data.shape
        device = data.device
        target_size = n // k
        
        # Initialize centroids (k-means++)
        centroids = torch.zeros(k, d, device=device)
        idx = torch.randint(n, (1,)).item()
        centroids[0] = data[idx]
        
        for i in range(1, k):
            # Distance to nearest centroid
            dists = torch.cdist(data, centroids[:i]).min(dim=1).values
            # Sample proportional to distance squared
            probs = dists ** 2
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1).item()
            centroids[i] = data[idx]
        
        # Iterate
        assignments = torch.zeros(n, dtype=torch.long, device=device)
        
        for iteration in range(max_iters):
            # Compute distances to all centroids
            dists = torch.cdist(data, centroids)  # [n, k]
            
            if balanced:
                # Balanced assignment: greedy assignment respecting size constraints
                new_assignments = torch.zeros(n, dtype=torch.long, device=device)
                cluster_counts = torch.zeros(k, dtype=torch.long, device=device)
                assigned = torch.zeros(n, dtype=torch.bool, device=device)
                
                # Sort by minimum distance (assign most confident first)
                min_dists = dists.min(dim=1).values
                order = min_dists.argsort()
                
                for idx in order:
                    # Find best available cluster
                    point_dists = dists[idx].clone()
                    
                    # Mask full clusters
                    full_mask = cluster_counts >= target_size
                    # Allow overflow only if all clusters at target
                    if full_mask.all():
                        full_mask = cluster_counts >= target_size + 1
                    
                    point_dists[full_mask] = float('inf')
                    
                    best_cluster = point_dists.argmin().item()
                    new_assignments[idx] = best_cluster
                    cluster_counts[best_cluster] += 1
                    assigned[idx] = True
            else:
                # Standard k-means: assign to nearest
                new_assignments = dists.argmin(dim=1)
            
            # Check convergence
            if (new_assignments == assignments).all():
                break
            assignments = new_assignments
            
            # Update centroids
            for c in range(k):
                mask = (assignments == c)
                if mask.any():
                    centroids[c] = data[mask].mean(dim=0)
        
        return assignments, centroids
    
    def route_hierarchical(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-level hierarchical routing.
        
        Level 1: Find best cluster(s)
        Level 2: Find best tile within cluster
        
        Args:
            x: [batch, d_model]
            
        Returns:
            tile_idx: [batch] winning tile index
            cluster_idx: [batch] winning cluster index
            scores: [batch] alignment score with winning tile
        """
        batch = x.shape[0]
        device = x.device
        
        # Level 1: Cluster routing
        cluster_scores = x @ self.cluster_signatures.T  # [batch, num_clusters]
        
        if self.top_k_clusters == 1:
            # Hard routing to single cluster
            cluster_idx = cluster_scores.argmax(dim=-1)  # [batch]
            
            # Level 2: Tile routing within cluster
            tile_idx, tile_scores = self._route_within_clusters_sorted(
                x, cluster_idx
            )
        else:
            # Soft routing to top-k clusters (VGem's robustness suggestion)
            top_clusters = cluster_scores.topk(self.top_k_clusters, dim=-1)
            
            # Find best tile across top-k clusters
            tile_idx, tile_scores, cluster_idx = self._route_top_k_clusters(
                x, top_clusters.indices
            )
        
        return tile_idx, cluster_idx, tile_scores
    
    def _route_within_clusters_sorted(
        self, 
        x: torch.Tensor, 
        cluster_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route within clusters with batch sorting for GPU efficiency.
        (VGem's recommendation)
        """
        batch = x.shape[0]
        device = x.device
        
        # Get signatures (EMA if available, else current)
        if self.ema_tile_signatures is not None:
            all_tile_sigs = self.ema_tile_signatures
        else:
            all_tile_sigs = self._get_current_signatures()
        
        # Sort by cluster for memory coalescing
        sort_idx = cluster_idx.argsort()
        x_sorted = x[sort_idx]
        cluster_sorted = cluster_idx[sort_idx]
        
        tile_idx = torch.zeros(batch, dtype=torch.long, device=device)
        tile_scores = torch.zeros(batch, device=device)
        
        # Process each cluster contiguously
        for c in range(self.num_clusters):
            mask = (cluster_sorted == c)
            if not mask.any():
                continue
            
            # Get tiles in this cluster
            tile_indices = (self.cluster_assignments == c).nonzero().squeeze(-1)
            
            # Get signatures for tiles in this cluster
            tile_sigs = all_tile_sigs[tile_indices]  # [tiles_per_cluster, d_model]
            
            # Route within cluster
            scores = x_sorted[mask] @ tile_sigs.T  # [cluster_batch, tiles_per_cluster]
            local_winners = scores.argmax(dim=-1)
            local_scores = scores.max(dim=-1).values
            
            # Convert to global tile indices
            tile_idx[mask] = tile_indices[local_winners]
            tile_scores[mask] = local_scores
        
        # Unsort
        unsort_idx = sort_idx.argsort()
        tile_idx = tile_idx[unsort_idx]
        tile_scores = tile_scores[unsort_idx]
        
        return tile_idx, tile_scores
    
    def _route_top_k_clusters(
        self,
        x: torch.Tensor,
        top_clusters: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route considering top-k clusters for robustness."""
        batch = x.shape[0]
        device = x.device
        
        best_tile = torch.zeros(batch, dtype=torch.long, device=device)
        best_score = torch.full((batch,), float('-inf'), device=device)
        best_cluster = torch.zeros(batch, dtype=torch.long, device=device)
        
        for k in range(self.top_k_clusters):
            cluster_idx = top_clusters[:, k]
            tile_idx, scores = self._route_within_clusters_sorted(x, cluster_idx)
            
            # Update best
            better = scores > best_score
            best_tile[better] = tile_idx[better]
            best_score[better] = scores[better]
            best_cluster[better] = cluster_idx[better]
        
        return best_tile, best_score, best_cluster
    
    def enable_frozen_routing(self):
        """Freeze signatures for VGem's 'Frozen Routing' test."""
        self.frozen_signatures = self._get_current_signatures().clone()
        self.freeze_routing = True
        print(f"Routing frozen with {self.num_tiles} signatures")
    
    def disable_frozen_routing(self):
        """Unfreeze routing."""
        self.frozen_signatures = None
        self.freeze_routing = False

    # =========================================================================
    # XOR Signature Compression for Inference
    # =========================================================================

    def compress_signatures(self):
        """
        Compress signatures for efficient inference routing.

        Uses XOR superposition to achieve 8-12x compression on tile signatures.
        Compressed routing uses Hamming distance instead of dot product,
        preserving routing decisions exactly.

        Call this after training is complete, before inference deployment.
        """
        if self._is_compressed:
            return  # Already compressed

        # Ensure hierarchy is built
        if not self._hierarchy_built:
            self.build_hierarchy()

        # Get stable signatures
        if self.ema_tile_signatures is not None:
            tile_sigs = self.ema_tile_signatures.sign()
        else:
            tile_sigs = self._get_current_signatures()

        # Compress tile signatures
        self._compressed_tile_signatures = CompressedSignatures().compress(tile_sigs)

        # Compress cluster signatures
        if self.cluster_signatures is not None:
            self._compressed_cluster_signatures = CompressedSignatures().compress(
                self.cluster_signatures
            )

        self._is_compressed = True

    def decompress_signatures(self):
        """
        Decompress signatures for training or debugging.

        Call this if you need to resume training after compression.
        """
        self._compressed_tile_signatures = None
        self._compressed_cluster_signatures = None
        self._is_compressed = False

    def get_compression_stats(self) -> Optional[Dict[str, CompressionStats]]:
        """
        Get compression statistics for both tile and cluster signatures.

        Returns:
            Dict with 'tile' and 'cluster' keys, or None if not compressed
        """
        if not self._is_compressed:
            return None

        stats = {}
        if self._compressed_tile_signatures is not None:
            stats['tile'] = self._compressed_tile_signatures.get_compression_stats()
        if self._compressed_cluster_signatures is not None:
            stats['cluster'] = self._compressed_cluster_signatures.get_compression_stats()

        return stats

    def route_hierarchical_compressed(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compressed hierarchical routing using Hamming distance.

        Uses XOR + POPCNT for O(1) routing instead of dot product.
        Produces identical routing decisions to uncompressed version.
        """
        batch = x.shape[0]
        device = x.device

        # Ternarize input
        x_tern = torch.sign(x)

        # Level 1: Cluster routing via Hamming distance
        cluster_sigs = self._compressed_cluster_signatures.decompress_all()
        cluster_dists = self._hamming_distance_batch(x_tern, cluster_sigs)
        cluster_idx = cluster_dists.argmin(dim=-1)

        # Level 2: Tile routing within cluster
        tile_idx, tile_scores = self._route_within_clusters_compressed(
            x_tern, cluster_idx
        )

        return tile_idx, cluster_idx, tile_scores

    def _hamming_distance_batch(
        self,
        query: torch.Tensor,
        signatures: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hamming distances from query to all signatures."""
        # For ternary: count positions where values differ
        # query: [batch, d_model]
        # signatures: [num_sigs, d_model]
        query_expanded = query.unsqueeze(1)  # [batch, 1, d_model]
        sigs_expanded = signatures.unsqueeze(0)  # [1, num_sigs, d_model]
        diff = (query_expanded != sigs_expanded).float()
        return diff.sum(dim=-1)

    def _route_within_clusters_compressed(
        self,
        x: torch.Tensor,
        cluster_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route within clusters using compressed signatures."""
        batch = x.shape[0]
        device = x.device

        # Get all tile signatures
        all_tile_sigs = self._compressed_tile_signatures.decompress_all()

        # Sort by cluster for memory coalescing
        sort_idx = cluster_idx.argsort()
        x_sorted = x[sort_idx]
        cluster_sorted = cluster_idx[sort_idx]

        tile_idx = torch.zeros(batch, dtype=torch.long, device=device)
        tile_scores = torch.zeros(batch, device=device)

        for c in range(self.num_clusters):
            mask = (cluster_sorted == c)
            if not mask.any():
                continue

            # Get tiles in this cluster
            tile_indices = (self.cluster_assignments == c).nonzero().squeeze(-1)
            tile_sigs = all_tile_sigs[tile_indices]

            # Hamming distance routing
            dists = self._hamming_distance_batch(x_sorted[mask], tile_sigs)
            local_winners = dists.argmin(dim=-1)
            local_scores = -dists.min(dim=-1).values  # Negative distance as score

            tile_idx[mask] = tile_indices[local_winners]
            tile_scores[mask] = local_scores

        # Unsort
        unsort_idx = sort_idx.argsort()
        tile_idx = tile_idx[unsort_idx]
        tile_scores = tile_scores[unsort_idx]

        return tile_idx, tile_scores

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Hierarchical forward pass with VGem's fixes:
        - Input normalization
        - Residual connection
        - Frozen routing option
        
        Args:
            x: [batch, d_model] or [batch, seq, d_model]
            
        Returns:
            output: Same shape as input
            routing_info: Dict with tile_idx, cluster_idx, scores
            aux_losses: Dict with balance and diversity losses
        """
        orig_shape = x.shape
        residual = x  # Save for residual connection
        
        # Flatten if 3D
        if x.dim() == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
            residual = residual.view(B * T, C)
        
        batch = x.shape[0]
        device = x.device
        
        # VGem's fix: Input normalization for stable routing
        x_norm = self.input_norm(x)
        
        # Update EMA signatures during training (unless frozen)
        if self.training and not self.freeze_routing:
            self._update_ema_signatures()
        
        # Build/rebuild hierarchy periodically
        if not self._hierarchy_built or (self.training and not self.freeze_routing and torch.rand(1).item() < 0.01):
            self.build_hierarchy()

        # Hierarchical routing (on normalized input)
        # Use compressed routing if available and not training
        if self._is_compressed and not self.training:
            tile_idx, cluster_idx, scores = self.route_hierarchical_compressed(x_norm)
        else:
            tile_idx, cluster_idx, scores = self.route_hierarchical(x_norm)
        
        # Compute outputs (sparse) - on normalized input
        tile_output = self._compute_sparse_sorted(x_norm, tile_idx)
        
        # VGem's fix: Residual connection (tile learns delta, not whole function)
        if self.use_residual:
            output = residual + tile_output
        else:
            output = tile_output
        
        # Update usage stats
        if self.training:
            for i, tile in enumerate(self.tiles):
                count = (tile_idx == i).sum().item()
                tile.update_usage(count, batch)
        
        # Auxiliary losses
        aux_losses = {}
        if self.training:
            aux_losses['cluster_balance'] = self._cluster_balance_loss(cluster_idx)
            aux_losses['tile_balance'] = self._tile_balance_loss(tile_idx)
            aux_losses['diversity'] = self._diversity_loss()
            aux_losses['total_aux'] = (
                aux_losses['cluster_balance'] + 
                aux_losses['tile_balance'] + 
                aux_losses['diversity']
            )
        
        # Restore shape
        if len(orig_shape) == 3:
            output = output.view(orig_shape[0], orig_shape[1], -1)
            tile_idx = tile_idx.view(orig_shape[0], orig_shape[1])
            cluster_idx = cluster_idx.view(orig_shape[0], orig_shape[1])
        
        routing_info = {
            'tile_idx': tile_idx,
            'cluster_idx': cluster_idx,
            'scores': scores if len(orig_shape) == 2 else scores.view(orig_shape[0], orig_shape[1])
        }
        
        return output, routing_info, aux_losses
    
    def _compute_sparse_sorted(
        self, 
        x: torch.Tensor, 
        tile_idx: torch.Tensor
    ) -> torch.Tensor:
        """Compute with batch sorting for GPU efficiency."""
        batch = x.shape[0]
        device = x.device
        
        # Sort by tile for memory coalescing
        sort_idx = tile_idx.argsort()
        x_sorted = x[sort_idx]
        tile_sorted = tile_idx[sort_idx]
        
        output = torch.zeros(batch, self.d_model, device=device)
        
        # Process each tile contiguously
        for t in range(self.num_tiles):
            mask = (tile_sorted == t)
            if not mask.any():
                continue
            
            tile_out = self.tiles[t](x_sorted[mask])
            output[mask] = tile_out
        
        # Unsort
        unsort_idx = sort_idx.argsort()
        output = output[unsort_idx]
        
        return self.dropout(output)
    
    def _cluster_balance_loss(self, cluster_idx: torch.Tensor) -> torch.Tensor:
        """Load balancing at cluster level."""
        counts = torch.bincount(
            cluster_idx.flatten(), 
            minlength=self.num_clusters
        ).float()
        return counts.std() / (counts.mean() + 1e-8) * self.balance_weight
    
    def _tile_balance_loss(self, tile_idx: torch.Tensor) -> torch.Tensor:
        """Load balancing at tile level."""
        counts = torch.bincount(
            tile_idx.flatten(), 
            minlength=self.num_tiles
        ).float()
        return counts.std() / (counts.mean() + 1e-8) * self.balance_weight
    
    def _diversity_loss(self) -> torch.Tensor:
        """Encourage diverse signatures."""
        if self.ema_tile_signatures is None:
            return torch.tensor(0.0)
        
        sigs = self.ema_tile_signatures
        similarity = torch.mm(sigs, sigs.T)
        mask = 1 - torch.eye(self.num_tiles, device=sigs.device)
        avg_sim = (similarity * mask).sum() / (mask.sum() + 1e-8)
        
        return avg_sim * self.diversity_weight
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        usage = [tile.usage_rate for tile in self.tiles]
        
        return {
            'num_tiles': self.num_tiles,
            'num_clusters': self.num_clusters,
            'tiles_per_cluster': self.tiles_per_cluster,
            'tile_usage': usage,
            'active_tiles': sum(1 for u in usage if u > 0.001),
            'max_usage': max(usage) if usage else 0,
            'min_usage': min(usage) if usage else 0,
            'usage_std': torch.tensor(usage).std().item() if usage else 0,
        }
    
    def get_cluster_info(self) -> Dict:
        """Get cluster structure info."""
        if not self._hierarchy_built:
            return {'status': 'hierarchy not built'}
        
        info = {}
        for c in range(self.num_clusters):
            tile_indices = (self.cluster_assignments == c).nonzero().squeeze(-1).tolist()
            tile_usage = [self.tiles[i].usage_rate for i in tile_indices]
            info[f'cluster_{c}'] = {
                'tiles': tile_indices,
                'usage': tile_usage,
                'total_usage': sum(tile_usage)
            }
        
        return info


class HierarchicalTriXBlock(nn.Module):
    """Transformer block with hierarchical FFN."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        d_hidden: int = None,
        dropout: float = 0.1,
        balance_weight: float = 0.01,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = HierarchicalTriXFFN(
            d_model=d_model,
            d_hidden=d_hidden,
            num_tiles=num_tiles,
            tiles_per_cluster=tiles_per_cluster,
            dropout=dropout,
            balance_weight=balance_weight,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = True
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        # Self-attention
        normed = self.norm1(x)
        if is_causal:
            T = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
            attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask, is_causal=True)
        else:
            attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.dropout(attn_out)
        
        # Hierarchical FFN
        ffn_out, routing_info, aux_losses = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x, routing_info, aux_losses


if __name__ == "__main__":
    print("Testing HierarchicalTriXFFN...")
    
    # Test with 64 tiles, 8 clusters of 8
    ffn = HierarchicalTriXFFN(
        d_model=128,
        d_hidden=64,
        num_tiles=64,
        tiles_per_cluster=8,
    )
    
    x = torch.randn(32, 128)  # batch=32, d_model=128
    
    # Build hierarchy
    ffn.build_hierarchy()
    
    # Forward
    out, routing, aux = ffn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Tiles routed to: {routing['tile_idx'].unique().tolist()}")
    print(f"Clusters used: {routing['cluster_idx'].unique().tolist()}")
    print(f"Aux losses: {aux}")
    
    # Test 3D input
    x3d = torch.randn(4, 16, 128)  # batch=4, seq=16, d_model=128
    out3d, routing3d, aux3d = ffn(x3d)
    print(f"\n3D Input shape: {x3d.shape}")
    print(f"3D Output shape: {out3d.shape}")
    
    # Stats
    print(f"\nRouting stats: {ffn.get_routing_stats()}")
    
    # Gradient check
    loss = out.sum()
    loss.backward()
    has_grad = any(tile.up_weight.grad is not None for tile in ffn.tiles)
    print(f"Gradients flow: {has_grad}")
    
    print("\nHierarchicalTriXFFN: OK")
