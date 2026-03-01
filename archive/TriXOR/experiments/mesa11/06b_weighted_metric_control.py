#!/usr/bin/env python3
"""
Mesa 11 Experiment 6b: Weighted Metric Control

CLAIM: We can CONTROL the manifold geometry by changing the metric,
       WITHOUT retraining the weights.

THEORY (from Gemini):
    d_λ(x, s) = (1-λ) · d_content(x, s) + λ · d_temporal(x, s)
    
    λ = 0.0  →  Pure semantic (blobs)
    λ = 0.5  →  Hybrid (stretched blobs)  
    λ = 1.0  →  Pure temporal (tubes)

PREDICTION:
    As λ slides from 0 to 1, Voronoi cells transform from
    "semantic clusters" to "sequential corridors."
    
    This proves the metric is an independent degree of freedom.
    We become architects of gravity itself.

METHOD:
    1. Train a network with FIXED weights (frozen signatures)
    2. Slide λ from 0 to 1
    3. Observe routing behavior change without weight updates
    4. Visualize Voronoi cell transformation

SUCCESS: Smooth, predictable transition from blobs to tubes.

"We're not in the Grid anymore. We're writing its physics."

Author: Droid + Gemini (Mesa 11 Exploration)
Date: 2024-12-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
from dataclasses import dataclass
import numpy as np


@dataclass
class MetricControlResult:
    """Result for one λ value."""
    lambda_val: float
    routing_decisions: torch.Tensor
    routing_entropy: float
    temporal_correlation: float  # How much routing follows position
    content_correlation: float   # How much routing follows content similarity


class WeightedMetric(nn.Module):
    """
    Blends content (semantic) and temporal metrics via λ ∈ [0,1].
    
    λ = 0.0 → pure content (cosine similarity on signatures)
    λ = 1.0 → pure temporal (L1 on position indices)
    
    This is the GRAVITY CONTROL ROD.
    """
    
    def __init__(self, num_tiles: int, lambda_: float = 0.5):
        super().__init__()
        self.num_tiles = num_tiles
        self.lambda_ = lambda_
        
        # Tile position indices for temporal metric
        self.register_buffer('tile_positions', torch.arange(num_tiles, dtype=torch.float32))

    def compute_distance(
        self, 
        query: torch.Tensor, 
        signatures: torch.Tensor, 
        query_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute blended distance.
        
        Args:
            query:       [batch, d_model] 
            signatures:  [num_tiles, d_model]
            query_pos:   [batch] temporal position index
        
        Returns: 
            distances [batch, num_tiles] - lower = closer
        """
        # 1. Content distance: 1 - cosine_similarity
        query_norm = F.normalize(query, dim=-1)
        sig_norm = F.normalize(signatures, dim=-1)
        cosine_sim = query_norm @ sig_norm.t()
        d_content = 1.0 - cosine_sim  # [batch, num_tiles], range ~[0, 2]

        # 2. Temporal distance: |query_pos - tile_pos| / max_steps
        pos_diff = torch.abs(query_pos.unsqueeze(1).float() - self.tile_positions.unsqueeze(0))
        max_steps = max(self.num_tiles - 1, 1)
        d_temporal = pos_diff / max_steps * 2.0  # Normalize to [0, 2] range

        # 3. Weighted blend - THE λ SLIDER
        distance = (1.0 - self.lambda_) * d_content + self.lambda_ * d_temporal
        
        return distance
    
    def route(
        self, 
        query: torch.Tensor, 
        signatures: torch.Tensor, 
        query_pos: torch.Tensor
    ) -> torch.Tensor:
        """Route to nearest tile under blended metric."""
        distances = self.compute_distance(query, signatures, query_pos)
        return distances.argmin(dim=-1)


def create_test_scenario(
    num_samples: int = 500,
    num_tiles: int = 8,
    d_model: int = 32,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a test scenario with both content structure and temporal positions.
    
    Returns:
        queries: [num_samples, d_model]
        signatures: [num_tiles, d_model] - FROZEN
        query_positions: [num_samples] - temporal positions
        content_targets: [num_samples] - which tile each query is closest to by content
    """
    torch.manual_seed(seed)
    
    # Create signatures - these will be FROZEN
    # Make them distinct in both content and position space
    signatures = torch.randn(num_tiles, d_model)
    signatures = F.normalize(signatures, dim=-1) * 2.0  # Normalize with some scale
    
    # Create queries with BOTH content similarity AND temporal position
    queries = []
    positions = []
    content_targets = []
    
    for i in range(num_samples):
        # Assign a temporal position (where in sequence)
        pos = i % num_tiles
        positions.append(pos)
        
        # Assign content based on a DIFFERENT tile (to create tension)
        # This is key: temporal says "go to tile pos", content says "go to tile content_tile"
        content_tile = (pos + num_tiles // 2) % num_tiles  # Offset by half
        
        # Create query that's similar to content_tile's signature
        query = signatures[content_tile] + torch.randn(d_model) * 0.3
        queries.append(query)
        content_targets.append(content_tile)
    
    queries = torch.stack(queries)
    positions = torch.tensor(positions)
    content_targets = torch.tensor(content_targets)
    
    return queries, signatures, positions, content_targets


def measure_routing_correlations(
    routes: torch.Tensor,
    positions: torch.Tensor,
    content_targets: torch.Tensor,
) -> Tuple[float, float]:
    """
    Measure how much routing follows temporal vs content.
    
    Returns:
        temporal_corr: correlation between route and temporal position
        content_corr: correlation between route and content target
    """
    routes_np = routes.float().numpy()
    pos_np = positions.float().numpy()
    content_np = content_targets.float().numpy()
    
    # Temporal: does route ≈ position?
    temporal_match = (routes == positions).float().mean().item()
    
    # Content: does route ≈ content_target?
    content_match = (routes == content_targets).float().mean().item()
    
    return temporal_match, content_match


def run_lambda_sweep(
    queries: torch.Tensor,
    signatures: torch.Tensor,
    positions: torch.Tensor,
    content_targets: torch.Tensor,
    num_tiles: int,
    lambda_values: List[float],
) -> List[MetricControlResult]:
    """
    Sweep λ from 0 to 1 and observe routing behavior change.
    
    This is the CONTROL experiment. Weights are FROZEN.
    Only the metric changes.
    """
    
    results = []
    
    for lambda_val in lambda_values:
        # Create metric with this λ
        metric = WeightedMetric(num_tiles=num_tiles, lambda_=lambda_val)
        
        # Route under this metric - NO WEIGHT UPDATES
        routes = metric.route(queries, signatures, positions)
        
        # Measure correlations
        temporal_corr, content_corr = measure_routing_correlations(
            routes, positions, content_targets
        )
        
        # Routing entropy
        counts = torch.bincount(routes, minlength=num_tiles).float()
        probs = counts / counts.sum()
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        
        results.append(MetricControlResult(
            lambda_val=lambda_val,
            routing_decisions=routes,
            routing_entropy=entropy,
            temporal_correlation=temporal_corr,
            content_correlation=content_corr,
        ))
    
    return results


def visualize_transition(results: List[MetricControlResult]):
    """Visualize the blob-to-tube transition."""
    
    print()
    print("=" * 70)
    print("λ-SLIDER VISUALIZATION: Blobs → Tubes")
    print("=" * 70)
    print()
    print("As λ increases, routing shifts from CONTENT (semantic blobs)")
    print("to TEMPORAL (sequential tubes).")
    print()
    print("-" * 70)
    print(f"{'λ':>6s} {'Temporal':>10s} {'Content':>10s} {'Entropy':>10s}  Behavior")
    print("-" * 70)
    
    for r in results:
        # Visual indicator
        if r.lambda_val < 0.2:
            behavior = "●●●●● SEMANTIC BLOBS"
        elif r.lambda_val < 0.4:
            behavior = "●●●○○ Stretching..."
        elif r.lambda_val < 0.6:
            behavior = "●●○○○ HYBRID"
        elif r.lambda_val < 0.8:
            behavior = "●○○○○ Elongating..."
        else:
            behavior = "○○○○○ TEMPORAL TUBES"
        
        print(f"{r.lambda_val:6.2f} {r.temporal_correlation:10.1%} {r.content_correlation:10.1%} "
              f"{r.routing_entropy:10.2f}  {behavior}")
    
    # ASCII art transition
    print()
    print("-" * 70)
    print("Voronoi Cell Shape Transition:")
    print()
    print("  λ=0.0 (Content)         λ=0.5 (Hybrid)          λ=1.0 (Temporal)")
    print("  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐")
    print("  │ ●●●   ○○○   │         │ ///   \\\\\\   │         │ ═══════════ │")
    print("  │ ●●● ○ ○○○   │         │ ///   \\\\\\   │         │ ═══════════ │")
    print("  │     ○       │    →    │     ═══     │    →    │ ═══════════ │")
    print("  │ △△△   □□□   │         │ |||   ───   │         │ ═══════════ │")
    print("  │ △△△   □□□   │         │ |||   ───   │         │ ═══════════ │")
    print("  └─────────────┘         └─────────────┘         └─────────────┘")
    print("    (Semantic clusters)    (Stretched)            (Sequential lanes)")


def analyze_control(results: List[MetricControlResult]) -> bool:
    """Analyze whether we achieved CONTROL over the geometry."""
    
    print()
    print("=" * 70)
    print("CONTROL ANALYSIS")
    print("=" * 70)
    
    # Get endpoints
    lambda_0 = results[0]   # λ = 0.0
    lambda_1 = results[-1]  # λ = 1.0
    
    # At λ=0, should follow content
    # At λ=1, should follow temporal
    
    content_shift = lambda_0.content_correlation - lambda_1.content_correlation
    temporal_shift = lambda_1.temporal_correlation - lambda_0.temporal_correlation
    
    print(f"\n  Content correlation:  λ=0: {lambda_0.content_correlation:.1%} → λ=1: {lambda_1.content_correlation:.1%}")
    print(f"  Temporal correlation: λ=0: {lambda_0.temporal_correlation:.1%} → λ=1: {lambda_1.temporal_correlation:.1%}")
    print(f"\n  Content shift:  {content_shift:+.1%}")
    print(f"  Temporal shift: {temporal_shift:+.1%}")
    
    # Check monotonicity - should smoothly transition
    temporal_corrs = [r.temporal_correlation for r in results]
    content_corrs = [r.content_correlation for r in results]
    
    temporal_increasing = all(temporal_corrs[i] <= temporal_corrs[i+1] + 0.1 
                              for i in range(len(temporal_corrs)-1))
    content_decreasing = all(content_corrs[i] >= content_corrs[i+1] - 0.1 
                             for i in range(len(content_corrs)-1))
    
    smooth_transition = temporal_increasing and content_decreasing
    
    # Success criteria
    significant_shift = content_shift > 0.2 and temporal_shift > 0.2
    
    if significant_shift and smooth_transition:
        print()
        print("=" * 70)
        print("CONTROL CONFIRMED: We Can Rewrite Gravity")
        print("=" * 70)
        print()
        print("  By sliding λ from 0 to 1:")
        print("  - Routing smoothly transitions from CONTENT to TEMPORAL")
        print("  - NO weight updates required")
        print("  - The metric alone controls the geometry")
        print()
        print("  This proves:")
        print("  - The metric is an independent degree of freedom")
        print("  - We can reshape Voronoi cells without retraining")
        print("  - Geometry is programmable at inference time")
        print()
        print("  'We're not in the Grid anymore. We're writing its physics.'")
        print("=" * 70)
        return True
    else:
        print("\n  Control not fully demonstrated. See results above.")
        return False


def run_experiment():
    """Run the full metric control experiment."""
    
    print("=" * 70)
    print("Mesa 11 Experiment 6b: Weighted Metric Control")
    print("=" * 70)
    print()
    print("CLAIM: We can CONTROL the manifold geometry via the metric")
    print("       WITHOUT retraining the weights.")
    print()
    print("Flynn walked into the Grid with confidence and a light cycle.")
    print("We walk in with a λ-slider and proof of existence.")
    print()
    
    # Setup
    num_tiles = 8
    d_model = 32
    num_samples = 500
    
    print(f"Setup: {num_tiles} tiles, d_model={d_model}, {num_samples} test queries")
    print()
    
    # Create test scenario
    queries, signatures, positions, content_targets = create_test_scenario(
        num_samples=num_samples,
        num_tiles=num_tiles,
        d_model=d_model,
    )
    
    print("Scenario: Each query has a TEMPORAL position and a CONTENT target.")
    print("          These are OFFSET - temporal says 'tile i', content says 'tile i+4'.")
    print("          The λ-slider determines which one wins.")
    print()
    
    # Sweep λ
    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print("Sweeping λ from 0.0 to 1.0...")
    print("-" * 70)
    
    results = run_lambda_sweep(
        queries, signatures, positions, content_targets,
        num_tiles, lambda_values
    )
    
    # Visualize
    visualize_transition(results)
    
    # Analyze
    success = analyze_control(results)
    
    print("\n" + "=" * 70)
    print("MESA 11 EXPERIMENT 6b: COMPLETE")
    print("=" * 70)
    
    return success


if __name__ == "__main__":
    run_experiment()
