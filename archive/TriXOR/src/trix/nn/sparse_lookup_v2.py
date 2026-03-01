"""
SparseLookupFFN v2: With Signature Surgery and Island Regularization

New capabilities:
  1. Signature surgery API (insert, freeze, unfreeze, claim tracking)
  2. Island-friendly regularizers (ternary pressure, sparsity, diversity)
  3. Score→gate spline calibration for routing stability
  
Based on findings from the Semantic Geometry Thesis experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List, Set
import math


# =============================================================================
# Score Calibration Spline (for routing stability)
# =============================================================================

class ScoreCalibrationSpline(nn.Module):
    """
    1D spline that calibrates routing scores → gates.
    
    Learns a monotonic-ish mapping that:
    - Sharpens confident scores
    - Softens ambiguous scores
    - Provides smooth gradients for training
    
    Input: raw score (unbounded)
    Output: gate value in [0, 1]
    """
    
    def __init__(self, num_knots: int = 8):
        super().__init__()
        self.num_knots = num_knots
        
        # Knot values (learnable)
        # Initialize as sigmoid-like: low→0, high→1
        init_vals = torch.sigmoid(torch.linspace(-3, 3, num_knots))
        self.knot_values = nn.Parameter(init_vals)
        
        # Temperature for score normalization
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: Raw routing scores [..., any shape]
        
        Returns:
            gates: Calibrated values in [0, 1], same shape
        """
        # Normalize scores to [0, 1] range using sigmoid
        normalized = torch.sigmoid(scores / (self.temperature + 1e-6))
        
        # Map to knot indices
        idx_float = normalized * (self.num_knots - 1)
        idx_low = idx_float.long().clamp(0, self.num_knots - 2)
        idx_high = (idx_low + 1).clamp(0, self.num_knots - 1)
        
        # Linear interpolation between knots
        t = idx_float - idx_low.float()
        
        val_low = self.knot_values[idx_low]
        val_high = self.knot_values[idx_high]
        
        output = val_low + t * (val_high - val_low)
        
        return output


# =============================================================================
# Enhanced SparseLookupFFN with Surgery and Regularization
# =============================================================================

class SparseLookupFFNv2(nn.Module):
    """
    SparseLookupFFN with surgery support and island regularization.
    
    New features:
      - Explicit signature parameters (not just derived from directions)
      - Signature surgery API: insert, freeze, unfreeze
      - Island regularizers: ternary pressure, sparsity, diversity
      - Score calibration splines for routing stability
      - Claim tracking for surgery validation
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        grid_size: int = 16,
        compress_hidden: Optional[int] = None,
        dropout: float = 0.1,
        # New options
        use_score_calibration: bool = True,
        ternary_weight: float = 0.01,
        sparsity_weight: float = 0.01,
        diversity_weight: float = 0.01,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_tiles = num_tiles
        self.num_clusters = num_tiles // tiles_per_cluster
        self.tiles_per_cluster = tiles_per_cluster
        self.grid_size = grid_size
        
        # Regularization weights
        self.ternary_weight = ternary_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        
        # ======================
        # EXPLICIT SIGNATURES (new)
        # ======================
        # Now separate from directions, allowing independent control
        self.signatures_raw = nn.Parameter(torch.randn(num_tiles, d_model) * 0.5)
        
        # Track frozen signatures
        self._frozen_tiles: Set[int] = set()
        self._surgery_history: List[Dict] = []
        
        # Score calibration spline (new)
        self.use_score_calibration = use_score_calibration
        if use_score_calibration:
            self.score_calibrator = ScoreCalibrationSpline(num_knots=8)
        
        # ======================
        # EXISTING COMPONENTS
        # ======================
        
        # Compression network
        compress_hidden = compress_hidden or d_model // 4
        self.compress = nn.Sequential(
            nn.Linear(d_model, compress_hidden),
            nn.GELU(),
            nn.Linear(compress_hidden, 2),
            nn.Tanh(),
        )
        
        # Per-tile magnitude splines
        self.magnitude_splines = nn.ModuleList([
            self._make_magnitude_spline(grid_size) for _ in range(num_tiles)
        ])
        
        # Tile directions (output contribution)
        self.directions = nn.Parameter(torch.randn(num_tiles, d_model) * 0.02)
        
        # Normalization and output
        self.norm = nn.LayerNorm(d_model)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.dropout = nn.Dropout(dropout)
        
        # Cluster assignments
        self.register_buffer(
            'cluster_assignments',
            torch.arange(num_tiles) // tiles_per_cluster
        )
        
        # Usage tracking
        self.register_buffer('tile_counts', torch.zeros(num_tiles))
        self.register_buffer('total_count', torch.tensor(0.0))
        
        # Claim tracking for surgery (new)
        self.register_buffer('claim_matrix', torch.zeros(num_tiles, num_tiles))
        # claim_matrix[i, j] = count of class j samples routed to tile i
    
    def _make_magnitude_spline(self, grid_size: int) -> nn.Module:
        """Create a magnitude spline for a tile."""
        return nn.Sequential(
            nn.Linear(2, grid_size),
            nn.ReLU(),
            nn.Linear(grid_size, 1),
        )
    
    # =========================================================================
    # SIGNATURE SURGERY API (new)
    # =========================================================================
    
    def _quantize_ternary(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to {-1, 0, +1} with straight-through estimator."""
        with torch.no_grad():
            q = torch.zeros_like(x)
            q[x > 0.3] = 1.0
            q[x < -0.3] = -1.0
        return x + (q - x).detach()
    
    @property
    def signatures(self) -> torch.Tensor:
        """Get quantized ternary signatures."""
        return self._quantize_ternary(self.signatures_raw)
    
    def insert_signature(
        self, 
        tile_idx: int, 
        signature: torch.Tensor,
        freeze: bool = True,
        tag: str = "",
    ) -> None:
        """
        Surgically insert a designed signature into a tile.
        
        Args:
            tile_idx: Which tile to modify
            signature: Ternary signature tensor [d_model]
            freeze: Whether to freeze the signature
            tag: Optional tag for tracking
        """
        assert 0 <= tile_idx < self.num_tiles
        assert signature.shape == (self.d_model,)
        
        with torch.no_grad():
            self.signatures_raw[tile_idx].copy_(signature)
        
        if freeze:
            self.freeze_signature(tile_idx)
        
        self._surgery_history.append({
            'action': 'insert',
            'tile_idx': tile_idx,
            'signature': signature.clone(),
            'frozen': freeze,
            'tag': tag,
        })
    
    def freeze_signature(self, tile_idx: int) -> None:
        """Freeze a tile's signature (no gradient updates)."""
        self._frozen_tiles.add(tile_idx)
    
    def unfreeze_signature(self, tile_idx: int) -> None:
        """Unfreeze a tile's signature."""
        self._frozen_tiles.discard(tile_idx)
    
    def is_frozen(self, tile_idx: int) -> bool:
        """Check if a tile's signature is frozen."""
        return tile_idx in self._frozen_tiles
    
    def get_surgery_history(self) -> List[Dict]:
        """Get history of surgery operations."""
        return self._surgery_history
    
    def get_signature_analysis(self, tile_idx: int) -> Dict:
        """Analyze a tile's current signature."""
        sig = self.signatures[tile_idx].detach()
        return {
            'tile_idx': tile_idx,
            'positive_dims': (sig > 0.5).nonzero(as_tuple=True)[0].tolist(),
            'negative_dims': (sig < -0.5).nonzero(as_tuple=True)[0].tolist(),
            'zero_count': ((sig > -0.5) & (sig < 0.5)).sum().item(),
            'frozen': self.is_frozen(tile_idx),
        }
    
    # =========================================================================
    # ROUTING WITH CALIBRATION
    # =========================================================================
    
    def get_cluster_signatures(self, signatures: torch.Tensor) -> torch.Tensor:
        """Compute cluster-level signatures."""
        cluster_sigs = []
        for c in range(self.num_clusters):
            mask = self.cluster_assignments == c
            cluster_sigs.append(signatures[mask].mean(dim=0).sign())
        return torch.stack(cluster_sigs)
    
    def route(
        self, 
        x: torch.Tensor, 
        signatures: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Hierarchical routing with optional score calibration.
        
        Returns:
            tile_idx: Selected tile per input [B*T]
            scores: Optional calibrated scores [B*T, num_tiles]
        """
        batch_size = x.shape[0]
        device = x.device
        
        cluster_sigs = self.get_cluster_signatures(signatures)
        
        # Level 1: Route to cluster
        cluster_scores = x @ cluster_sigs.T
        
        if self.use_score_calibration:
            cluster_gates = self.score_calibrator(cluster_scores)
            cluster_idx = cluster_gates.argmax(dim=-1)
        else:
            cluster_idx = cluster_scores.argmax(dim=-1)
        
        # Level 2: Route to tile within cluster
        tile_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        all_scores = torch.zeros(batch_size, self.num_tiles, device=device) if return_scores else None
        
        for c in range(self.num_clusters):
            mask = cluster_idx == c
            if not mask.any():
                continue
            
            tile_mask = self.cluster_assignments == c
            cluster_tiles = torch.where(tile_mask)[0]
            cluster_tile_sigs = signatures[tile_mask]
            
            scores = x[mask] @ cluster_tile_sigs.T
            
            if self.use_score_calibration:
                gates = self.score_calibrator(scores)
                local_idx = gates.argmax(dim=-1)
            else:
                local_idx = scores.argmax(dim=-1)
            
            tile_idx[mask] = cluster_tiles[local_idx]
            
            if return_scores:
                all_scores[mask][:, cluster_tiles] = scores
        
        return tile_idx, all_scores
    
    # =========================================================================
    # ISLAND REGULARIZERS (new)
    # =========================================================================
    
    def compute_ternary_loss(self) -> torch.Tensor:
        """
        Encourage signatures to be close to {-1, 0, +1}.
        
        Loss = mean distance to nearest ternary value.
        """
        sigs = self.signatures_raw
        
        # Distance to nearest ternary value
        dist_to_neg1 = (sigs - (-1)).abs()
        dist_to_zero = sigs.abs()
        dist_to_pos1 = (sigs - 1).abs()
        
        min_dist = torch.min(torch.min(dist_to_neg1, dist_to_zero), dist_to_pos1)
        
        return min_dist.mean()
    
    def compute_sparsity_loss(self) -> torch.Tensor:
        """
        Encourage sparse signatures (many zeros).
        
        Penalize signatures where too many dimensions are non-zero.
        """
        sigs = self.signatures
        
        # Count non-zeros per signature
        nonzero_count = (sigs.abs() > 0.5).float().sum(dim=-1)
        
        # Ideal: ~30% non-zero (adjustable)
        target_nonzero = 0.3 * self.d_model
        
        # Penalize if too many non-zeros
        excess = F.relu(nonzero_count - target_nonzero)
        
        return excess.mean() / self.d_model
    
    def compute_diversity_loss(self) -> torch.Tensor:
        """
        Encourage diverse signatures (tiles should be different).
        
        Penalize high cosine similarity between signature pairs.
        """
        sigs = self.signatures
        
        # Normalize
        sigs_norm = F.normalize(sigs, dim=-1)
        
        # Pairwise similarity
        sim_matrix = sigs_norm @ sigs_norm.T
        
        # Mask diagonal (self-similarity)
        mask = ~torch.eye(self.num_tiles, dtype=torch.bool, device=sigs.device)
        
        # Penalize high similarity
        high_sim = F.relu(sim_matrix[mask] - 0.5)  # Penalize >0.5 similarity
        
        return high_sim.mean()
    
    # =========================================================================
    # FORWARD PASS
    # =========================================================================
    
    def forward(
        self, 
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,  # For claim tracking
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Forward pass with optional claim tracking.
        
        Args:
            x: Input tensor [B, T, d_model]
            labels: Optional class labels [B, T] for claim tracking
        
        Returns:
            output: Output tensor [B, T, d_model]
            routing_info: Dict with routing details
            aux_losses: Dict with auxiliary losses
        """
        B, T, D = x.shape
        device = x.device
        
        # Apply frozen masks to gradients
        if self._frozen_tiles and self.training:
            # Zero out gradients for frozen tiles after backward
            for tile_idx in self._frozen_tiles:
                self.signatures_raw.grad_fn  # Ensure in computation graph
        
        # Normalize
        x_norm = self.norm(x)
        x_flat = x_norm.view(-1, D)
        
        # Get signatures
        signatures = self.signatures
        
        # Route
        tile_idx, scores = self.route(x_flat, signatures, return_scores=True)
        
        # Compress
        compressed = self.compress(x_flat)
        
        # Sparse lookup with magnitude splines
        output = torch.zeros_like(x_flat)
        
        for t in range(self.num_tiles):
            mask = tile_idx == t
            if not mask.any():
                continue
            
            # Magnitude spline
            scale = self.magnitude_splines[t](compressed[mask]).squeeze(-1)
            
            # Apply direction
            output[mask] = scale.unsqueeze(-1) * self.directions[t]
            
            # Track usage
            if self.training:
                self.tile_counts[t] += mask.sum().float()
        
        if self.training:
            self.total_count += B * T
        
        # Track claims if labels provided
        if labels is not None and self.training:
            self._update_claim_matrix(tile_idx, labels.view(-1))
        
        # Reshape and output
        output = output.view(B, T, D) * self.output_scale
        output = self.dropout(output)
        output = x + output  # Residual
        
        # Routing info
        routing_info = {
            'tile_idx': tile_idx.view(B, T),
            'compressed': compressed.view(B, T, 2),
        }
        
        # Compute all losses
        aux_losses = self._compute_aux_losses(tile_idx, B * T)
        
        return output, routing_info, aux_losses
    
    def _update_claim_matrix(self, tile_idx: torch.Tensor, labels: torch.Tensor) -> None:
        """Update claim tracking matrix."""
        for t in range(self.num_tiles):
            mask = tile_idx == t
            if mask.any():
                tile_labels = labels[mask]
                for label in tile_labels.unique():
                    count = (tile_labels == label).sum()
                    self.claim_matrix[t, label] += count.float()
    
    def _compute_aux_losses(self, tile_idx: torch.Tensor, total: int) -> Dict:
        """Compute all auxiliary losses."""
        device = tile_idx.device
        
        # Balance loss
        counts = torch.zeros(self.num_tiles, device=device)
        for t in range(self.num_tiles):
            counts[t] = (tile_idx == t).sum().float()
        ideal = total / self.num_tiles
        balance_loss = ((counts - ideal) ** 2).mean() / (ideal ** 2 + 1e-8)
        
        # Island regularizers
        ternary_loss = self.compute_ternary_loss() if self.ternary_weight > 0 else torch.tensor(0.0, device=device)
        sparsity_loss = self.compute_sparsity_loss() if self.sparsity_weight > 0 else torch.tensor(0.0, device=device)
        diversity_loss = self.compute_diversity_loss() if self.diversity_weight > 0 else torch.tensor(0.0, device=device)
        
        total_aux = (
            balance_loss * 0.01 +
            ternary_loss * self.ternary_weight +
            sparsity_loss * self.sparsity_weight +
            diversity_loss * self.diversity_weight
        )
        
        return {
            'balance': balance_loss * 0.01,
            'ternary': ternary_loss * self.ternary_weight,
            'sparsity': sparsity_loss * self.sparsity_weight,
            'diversity': diversity_loss * self.diversity_weight,
            'total_aux': total_aux,
        }
    
    # =========================================================================
    # SURGERY VALIDATION
    # =========================================================================
    
    def get_claim_rate(self, tile_idx: int, target_class: int) -> float:
        """
        Get claim rate for a surgically inserted tile.
        
        Returns: fraction of target_class samples that route to tile_idx
        """
        if self.claim_matrix.sum() == 0:
            return 0.0
        
        tile_claims = self.claim_matrix[tile_idx, target_class]
        total_class = self.claim_matrix[:, target_class].sum()
        
        if total_class == 0:
            return 0.0
        
        return (tile_claims / total_class).item()
    
    def get_tile_purity(self, tile_idx: int) -> Tuple[int, float]:
        """
        Get purity of a tile (what class does it mostly handle?).
        
        Returns: (dominant_class, purity)
        """
        tile_counts = self.claim_matrix[tile_idx]
        total = tile_counts.sum()
        
        if total == 0:
            return -1, 0.0
        
        dominant_class = tile_counts.argmax().item()
        purity = (tile_counts[dominant_class] / total).item()
        
        return dominant_class, purity
    
    def reset_claim_tracking(self) -> None:
        """Reset claim tracking matrix."""
        self.claim_matrix.zero_()
    
    # =========================================================================
    # STATS AND UTILS
    # =========================================================================
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        if self.total_count == 0:
            return {'num_tiles': self.num_tiles, 'active_tiles': 0}
        
        usage = self.tile_counts / self.total_count
        
        return {
            'num_tiles': self.num_tiles,
            'num_clusters': self.num_clusters,
            'active_tiles': (usage > 0.001).sum().item(),
            'frozen_tiles': len(self._frozen_tiles),
            'usage_mean': usage.mean().item(),
            'usage_std': usage.std().item(),
        }
    
    def get_island_stats(self) -> Dict:
        """Get statistics about signature quality (island-ness)."""
        sigs = self.signatures.detach()
        
        # Ternary-ness
        near_ternary = ((sigs.abs() < 0.2) | (sigs.abs() > 0.8)).float().mean().item()
        
        # Sparsity
        sparsity = (sigs.abs() < 0.2).float().mean().item()
        
        # Diversity (mean pairwise distance)
        sigs_norm = F.normalize(sigs, dim=-1)
        sim = (sigs_norm @ sigs_norm.T)
        mask = ~torch.eye(self.num_tiles, dtype=torch.bool, device=sigs.device)
        mean_sim = sim[mask].mean().item()
        
        return {
            'ternary_fraction': near_ternary,
            'sparsity': sparsity,
            'mean_pairwise_similarity': mean_sim,
            'diversity': 1 - mean_sim,
        }
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.tile_counts.zero_()
        self.total_count.zero_()
        self.claim_matrix.zero_()


# =============================================================================
# Block wrapper
# =============================================================================

class SparseLookupBlockV2(nn.Module):
    """Transformer block using SparseLookupFFNv2."""
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        dropout: float = 0.1,
        **ffn_kwargs,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.ffn = SparseLookupFFNv2(
            d_model=d_model,
            num_tiles=num_tiles,
            tiles_per_cluster=tiles_per_cluster,
            dropout=dropout,
            **ffn_kwargs,
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        is_causal: bool = True,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        B, T, D = x.shape
        
        # Attention
        x_norm = self.ln1(x)
        if is_causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        x = x + self.dropout(attn_out)
        
        # FFN
        x, routing_info, aux_losses = self.ffn(x, labels=labels)
        
        return x, routing_info, aux_losses


# =============================================================================
# Test
# =============================================================================

def test_sparse_lookup_v2():
    """Test SparseLookupFFNv2 with surgery."""
    print("=" * 70)
    print("SparseLookupFFNv2 Test (with Surgery + Regularization)")
    print("=" * 70)
    
    d_model = 64
    num_tiles = 16
    
    model = SparseLookupFFNv2(
        d_model=d_model,
        num_tiles=num_tiles,
        tiles_per_cluster=4,
        ternary_weight=0.01,
        sparsity_weight=0.01,
        diversity_weight=0.01,
    )
    
    # Test forward pass
    x = torch.randn(2, 16, d_model)
    labels = torch.randint(0, 10, (2, 16))
    
    output, routing_info, aux_losses = model(x, labels=labels)
    
    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Aux losses: {list(aux_losses.keys())}")
    
    # Test surgery
    print(f"\n--- Surgery Test ---")
    
    # Design a signature
    designed_sig = torch.zeros(d_model)
    designed_sig[:8] = 1.0  # Positive on first 8 dims
    
    # Insert
    model.insert_signature(0, designed_sig, freeze=True, tag="test_class_0")
    
    print(f"  Inserted signature into tile 0")
    print(f"  Frozen tiles: {model._frozen_tiles}")
    
    # Check analysis
    analysis = model.get_signature_analysis(0)
    print(f"  Tile 0 analysis: {analysis}")
    
    # Forward with labels
    for _ in range(10):
        output, routing_info, aux_losses = model(x, labels=labels)
    
    # Check island stats
    island_stats = model.get_island_stats()
    print(f"\n  Island stats: {island_stats}")
    
    # Unfreeze and check stability
    model.unfreeze_signature(0)
    print(f"\n  Unfroze tile 0")
    
    # More forward passes
    for _ in range(10):
        output, routing_info, aux_losses = model(x, labels=labels)
        loss = output.sum() + aux_losses['total_aux']
        loss.backward()
    
    # Check if signature drifted
    new_analysis = model.get_signature_analysis(0)
    print(f"  Tile 0 after training: {new_analysis}")
    
    print("\n" + "=" * 70)
    print("Test PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_sparse_lookup_v2()
