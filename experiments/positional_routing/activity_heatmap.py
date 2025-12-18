#!/usr/bin/env python3
"""
Activity Heatmap: Visualizing the Geometry of Routing

The Test:
- Without positional routing: "sparkle" pattern (random access)
- With positional routing: "wave" pattern (diagonal structure)

If we see the wave, we've encoded the geometry of time into compute topology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict, Optional


def cubic_bspline(t: torch.Tensor) -> torch.Tensor:
    """Cubic B-spline kernel. C² continuous."""
    t = t.abs()
    result = torch.zeros_like(t)
    
    mask1 = t < 1
    result[mask1] = (2/3) - t[mask1]**2 + 0.5 * t[mask1]**3
    
    mask2 = (t >= 1) & (t < 2)
    result[mask2] = (1/6) * (2 - t[mask2])**3
    
    return result


class ContentOnlyRouter(nn.Module):
    """Standard MoE routing - content only, no position."""
    
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        # Expert signatures (ternary-style)
        self.signatures = nn.Parameter(torch.randn(num_experts, d_model))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D]
        Returns:
            expert_idx: [B, T] - which expert
            scores: [B, T, num_experts] - routing scores
        """
        # Content-only routing
        sigs = self.signatures.sign()  # Ternary
        scores = torch.einsum('btd,ed->bte', x, sigs)
        expert_idx = scores.argmax(dim=-1)
        return expert_idx, F.softmax(scores, dim=-1)


class PositionalRouter(nn.Module):
    """Positional routing - content × position (B-spline spreading)."""
    
    def __init__(self, d_model: int, num_experts: int, max_seq_len: int, spread: float = 2.0):
        super().__init__()
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len
        self.spread = spread
        
        # Expert signatures (content)
        self.signatures = nn.Parameter(torch.randn(num_experts, d_model))
        
        # Expert positions (evenly spaced along sequence)
        expert_positions = torch.linspace(0, max_seq_len, num_experts)
        self.register_buffer('expert_positions', expert_positions)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D]
            positions: [B, T] or None (defaults to 0, 1, 2, ...)
        Returns:
            expert_idx: [B, T]
            scores: [B, T, num_experts]
        """
        B, T, D = x.shape
        device = x.device
        
        if positions is None:
            positions = torch.arange(T, device=device).float().unsqueeze(0).expand(B, -1)
        
        # Content scores
        sigs = self.signatures.sign()
        content_scores = torch.einsum('btd,ed->bte', x, sigs)
        
        # Position scores (B-spline spreading)
        # Normalize positions to expert space
        pos_normalized = positions / self.max_seq_len * self.num_experts
        expert_centers = torch.arange(self.num_experts, device=device).float()
        
        # Distance from each position to each expert
        pos_diff = pos_normalized.unsqueeze(-1) - expert_centers.unsqueeze(0).unsqueeze(0)
        position_scores = cubic_bspline(pos_diff / self.spread)
        
        # Combined: content × position
        combined_scores = content_scores * position_scores
        
        # Soft routing (for visualization)
        soft_scores = F.softmax(combined_scores, dim=-1)
        expert_idx = combined_scores.argmax(dim=-1)
        
        return expert_idx, soft_scores


def generate_activity_heatmap(router: nn.Module, seq_len: int = 128, d_model: int = 64, 
                               num_samples: int = 100) -> np.ndarray:
    """
    Generate activity heatmap: which experts fire at which positions.
    
    Returns:
        heatmap: [num_experts, seq_len] - average routing weight
    """
    router.eval()
    device = next(router.parameters()).device
    
    heatmap = torch.zeros(router.num_experts, seq_len, device=device)
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Random input
            x = torch.randn(1, seq_len, d_model, device=device)
            
            # Get routing scores
            if isinstance(router, PositionalRouter):
                _, scores = router(x)
            else:
                _, scores = router(x)
            
            # Accumulate: scores is [1, T, num_experts]
            heatmap += scores[0].T  # [num_experts, T]
    
    heatmap /= num_samples
    return heatmap.cpu().numpy()


def print_heatmap(heatmap: np.ndarray, title: str, width: int = 80):
    """Print ASCII heatmap to terminal."""
    num_experts, seq_len = heatmap.shape
    
    # Downsample for display
    display_experts = min(num_experts, 20)
    display_positions = min(seq_len, width - 10)
    
    expert_stride = max(1, num_experts // display_experts)
    pos_stride = max(1, seq_len // display_positions)
    
    sampled = heatmap[::expert_stride, ::pos_stride]
    
    # Normalize to 0-1
    sampled = (sampled - sampled.min()) / (sampled.max() - sampled.min() + 1e-8)
    
    # Characters by intensity
    chars = ' ·:+*#@'
    
    print()
    print('=' * 70)
    print(title)
    print('=' * 70)
    print(f'  Experts (rows) vs Positions (columns)')
    print(f'  Intensity: [ ] low → [@] high')
    print()
    
    # Y-axis label
    print(f'{"E":>3}|', end='')
    
    # Print heatmap
    for i, row in enumerate(sampled):
        if i > 0:
            print(f'{i*expert_stride:>3}|', end='')
        for val in row:
            char_idx = int(val * (len(chars) - 1))
            print(chars[char_idx], end='')
        print()
    
    # X-axis
    print('   +' + '-' * len(sampled[0]))
    print('    Position →')
    print()


def measure_diagonality(heatmap: np.ndarray) -> float:
    """
    Measure how "diagonal" the heatmap is.
    
    A perfect diagonal wave would have high correlation between
    expert index and position.
    
    Returns:
        score: 0 = random/sparkle, 1 = perfect diagonal
    """
    num_experts, seq_len = heatmap.shape
    
    # For each position, find the center of mass of expert activation
    expert_indices = np.arange(num_experts)
    
    centers = []
    for t in range(seq_len):
        weights = heatmap[:, t]
        if weights.sum() > 0:
            center = np.average(expert_indices, weights=weights)
            centers.append(center)
    
    centers = np.array(centers)
    positions = np.arange(len(centers))
    
    # Expected center if perfectly diagonal
    expected = positions / seq_len * num_experts
    
    # Correlation
    if len(centers) > 1:
        correlation = np.corrcoef(centers, expected)[0, 1]
        return max(0, correlation)  # Clamp negative correlations
    return 0.0


def main():
    print('=' * 70)
    print('ACTIVITY HEATMAP: Geometry of Routing')
    print('=' * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Parameters
    d_model = 64
    num_experts = 16
    seq_len = 64
    max_seq_len = 128
    
    print(f'Model: d={d_model}, experts={num_experts}, seq_len={seq_len}')
    print()
    
    # Content-only router (standard MoE)
    content_router = ContentOnlyRouter(d_model, num_experts).to(device)
    
    # Positional router (our proposal)
    positional_router = PositionalRouter(d_model, num_experts, max_seq_len, spread=2.0).to(device)
    
    # Generate heatmaps
    print('Generating heatmaps (100 samples each)...')
    
    heatmap_content = generate_activity_heatmap(content_router, seq_len, d_model)
    heatmap_positional = generate_activity_heatmap(positional_router, seq_len, d_model)
    
    # Display
    print_heatmap(heatmap_content, 'CONTENT-ONLY ROUTING (Standard MoE)')
    diag_content = measure_diagonality(heatmap_content)
    print(f'  Diagonality score: {diag_content:.3f} (0=sparkle, 1=wave)')
    
    print_heatmap(heatmap_positional, 'POSITIONAL ROUTING (B-spline × Content)')
    diag_positional = measure_diagonality(heatmap_positional)
    print(f'  Diagonality score: {diag_positional:.3f} (0=sparkle, 1=wave)')
    
    # Verdict
    print()
    print('=' * 70)
    print('VERDICT')
    print('=' * 70)
    
    if diag_positional > diag_content + 0.3:
        print('✓ WAVE DETECTED!')
        print('  Positional routing encodes temporal geometry.')
        print('  The "tube" structure is visible.')
    elif diag_positional > diag_content + 0.1:
        print('~ PARTIAL WAVE')
        print('  Some temporal structure visible.')
        print('  May need tuning (spread parameter, expert count).')
    else:
        print('✗ NO CLEAR DIFFERENCE')
        print('  Positional routing may not be helping here.')
        print('  Check: Is content already encoding position?')
    
    print()
    print(f'Diagonality improvement: {diag_positional - diag_content:+.3f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
