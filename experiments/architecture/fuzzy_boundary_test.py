#!/usr/bin/env python3
"""
Fuzzy Boundary Test: Does the mechanism degrade gracefully under overlap?

Two tests:
  1) Shared-dimension overlap - classes share 1-2 dims
  2) Gradient boundaries - soft class membership

Pass condition: Purity drops (expected) but specialization still emerges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Model (same as convergence_test.py)
# =============================================================================

class MinimalTriXLayer(nn.Module):
    def __init__(self, d_model: int, n_tiles: int, n_classes: int):
        super().__init__()
        self.d_model = d_model
        self.n_tiles = n_tiles
        self.n_classes = n_classes
        self.signatures_raw = nn.Parameter(torch.randn(n_tiles, d_model) * 0.5)
        self.tile_outputs = nn.ModuleList([
            nn.Linear(d_model, n_classes) for _ in range(n_tiles)
        ])
    
    def _quantize_ternary(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = torch.zeros_like(x)
            q[x > 0.3] = 1.0
            q[x < -0.3] = -1.0
        return x + (q - x).detach()
    
    @property
    def signatures(self) -> torch.Tensor:
        return self._quantize_ternary(self.signatures_raw)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        sigs = self.signatures
        scores = x @ sigs.T
        tile_indices = scores.argmax(dim=-1)
        
        logits = torch.zeros(batch_size, self.n_classes, device=x.device)
        for tile_idx in range(self.n_tiles):
            mask = tile_indices == tile_idx
            if mask.any():
                logits[mask] = self.tile_outputs[tile_idx](x[mask])
        
        return logits, tile_indices
    
    def compute_routing_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sigs = self.signatures_raw
        scores = x @ sigs.T
        log_probs = F.log_softmax(scores, dim=-1)
        if self.n_tiles == self.n_classes:
            return F.nll_loss(log_probs, y)
        return torch.tensor(0.0, device=x.device)


# =============================================================================
# Test 1: Shared-Dimension Overlap
# =============================================================================

@dataclass
class OverlappingClass:
    name: str
    positive_dims: list


# Classes now SHARE dimensions
OVERLAPPING_CLASSES = [
    OverlappingClass("A", positive_dims=[0, 1, 2]),      # unique: 0,1; shared: 2
    OverlappingClass("B", positive_dims=[2, 3, 4]),      # unique: 3,4; shared: 2
    OverlappingClass("C", positive_dims=[4, 5, 6]),      # unique: 5,6; shared: 4
    OverlappingClass("D", positive_dims=[6, 7, 8]),      # unique: 7,8; shared: 6
    OverlappingClass("E", positive_dims=[8, 9, 10]),     # unique: 9,10; shared: 8
]


def generate_overlapping_data(
    n_samples: int,
    d_model: int = 16,
    signal_strength: float = 1.0,
    noise_scale: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate data with overlapping class dimensions."""
    n_classes = len(OVERLAPPING_CLASSES)
    samples_per_class = n_samples // n_classes
    
    xs, ys = [], []
    
    for class_idx, cls in enumerate(OVERLAPPING_CLASSES):
        x = torch.randn(samples_per_class, d_model) * noise_scale
        for dim in cls.positive_dims:
            x[:, dim] = signal_strength + torch.randn(samples_per_class) * noise_scale
        xs.append(x)
        ys.append(torch.full((samples_per_class,), class_idx, dtype=torch.long))
    
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    perm = torch.randperm(len(x))
    return x[perm], y[perm]


def run_overlap_test(
    n_epochs: int = 100,
    n_train: int = 1000,
    n_test: int = 500,
    d_model: int = 16,
    seed: int = 42,
) -> dict:
    """Test with shared-dimension overlap."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_classes = len(OVERLAPPING_CLASSES)
    n_tiles = n_classes
    
    x_train, y_train = generate_overlapping_data(n_train, d_model)
    x_test, y_test = generate_overlapping_data(n_test, d_model)
    
    model = MinimalTriXLayer(d_model, n_tiles, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(n_epochs):
        model.train()
        logits, _ = model(x_train)
        task_loss = F.cross_entropy(logits, y_train)
        routing_loss = model.compute_routing_loss(x_train, y_train)
        loss = task_loss + routing_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_logits, tile_indices = model(x_test)
        test_acc = (test_logits.argmax(-1) == y_test).float().mean().item()
    
    # Compute purity per class
    purity_per_class = {}
    for class_idx, cls in enumerate(OVERLAPPING_CLASSES):
        mask = y_test == class_idx
        class_tiles = tile_indices[mask]
        if len(class_tiles) > 0:
            dominant = class_tiles.mode().values.item()
            purity = (class_tiles == dominant).float().mean().item()
            purity_per_class[cls.name] = {"dominant_tile": dominant, "purity": purity}
    
    mean_purity = np.mean([v["purity"] for v in purity_per_class.values()])
    
    # Check signature specialization
    sigs = model.signatures.detach()
    specialization = {}
    for tile_idx in range(n_tiles):
        sig = sigs[tile_idx]
        pos_dims = (sig > 0.5).nonzero(as_tuple=True)[0].tolist()
        neg_dims = (sig < -0.5).nonzero(as_tuple=True)[0].tolist()
        specialization[tile_idx] = {"pos": pos_dims, "neg": neg_dims}
    
    return {
        "test_accuracy": test_acc,
        "mean_purity": mean_purity,
        "purity_per_class": purity_per_class,
        "specialization": specialization,
    }


# =============================================================================
# Test 2: Gradient Boundaries (Soft Labels)
# =============================================================================

def generate_gradient_data(
    n_samples: int,
    d_model: int = 16,
    n_classes: int = 5,
    signal_strength: float = 1.0,
    noise_scale: float = 0.3,
    mixing_prob: float = 0.2,  # Probability of mixing with adjacent class
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data with gradient/soft boundaries.
    Some samples are mixtures of adjacent classes.
    """
    samples_per_class = n_samples // n_classes
    
    # Define class centers (use first few dims for each class)
    class_dims = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, 14],
    ]
    
    xs, ys = [], []
    
    for class_idx in range(n_classes):
        for _ in range(samples_per_class):
            x = torch.randn(d_model) * noise_scale
            
            # Pure class sample or mixture?
            if torch.rand(1).item() < mixing_prob and class_idx < n_classes - 1:
                # Mixture: interpolate with next class
                mix_ratio = torch.rand(1).item()  # 0 to 1
                
                # Primary class dims
                for dim in class_dims[class_idx]:
                    x[dim] = signal_strength * (1 - mix_ratio) + torch.randn(1).item() * noise_scale
                
                # Secondary class dims  
                for dim in class_dims[class_idx + 1]:
                    x[dim] = signal_strength * mix_ratio + torch.randn(1).item() * noise_scale
                
                # Label is still primary class (but boundary is fuzzy)
            else:
                # Pure sample
                for dim in class_dims[class_idx]:
                    x[dim] = signal_strength + torch.randn(1).item() * noise_scale
            
            xs.append(x)
            ys.append(class_idx)
    
    x = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    perm = torch.randperm(len(x))
    return x[perm], y[perm]


def run_gradient_test(
    n_epochs: int = 100,
    n_train: int = 1000,
    n_test: int = 500,
    d_model: int = 16,
    mixing_prob: float = 0.2,
    seed: int = 42,
) -> dict:
    """Test with gradient/soft boundaries."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_classes = 5
    n_tiles = n_classes
    
    x_train, y_train = generate_gradient_data(n_train, d_model, n_classes, mixing_prob=mixing_prob)
    x_test, y_test = generate_gradient_data(n_test, d_model, n_classes, mixing_prob=mixing_prob)
    
    model = MinimalTriXLayer(d_model, n_tiles, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(n_epochs):
        model.train()
        logits, _ = model(x_train)
        task_loss = F.cross_entropy(logits, y_train)
        routing_loss = model.compute_routing_loss(x_train, y_train)
        loss = task_loss + routing_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_logits, tile_indices = model(x_test)
        test_acc = (test_logits.argmax(-1) == y_test).float().mean().item()
    
    # Compute purity
    purity_per_class = {}
    for class_idx in range(n_classes):
        mask = y_test == class_idx
        class_tiles = tile_indices[mask]
        if len(class_tiles) > 0:
            dominant = class_tiles.mode().values.item()
            purity = (class_tiles == dominant).float().mean().item()
            purity_per_class[f"Class_{class_idx}"] = {"dominant_tile": dominant, "purity": purity}
    
    mean_purity = np.mean([v["purity"] for v in purity_per_class.values()])
    
    # Signature analysis
    sigs = model.signatures.detach()
    specialization = {}
    for tile_idx in range(n_tiles):
        sig = sigs[tile_idx]
        pos_dims = (sig > 0.5).nonzero(as_tuple=True)[0].tolist()
        specialization[tile_idx] = pos_dims
    
    return {
        "test_accuracy": test_acc,
        "mean_purity": mean_purity,
        "purity_per_class": purity_per_class,
        "specialization": specialization,
        "mixing_prob": mixing_prob,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("FUZZY BOUNDARY TEST: Graceful degradation under overlap")
    print("=" * 70)
    
    # Baseline reference (clean data)
    print("\n[BASELINE] Clean orthogonal classes:")
    from convergence_test import run_convergence_test
    # Quick single-seed baseline
    torch.manual_seed(42)
    from convergence_test import train_single_seed
    baseline = train_single_seed(42)
    print(f"    Accuracy: {baseline['test_accuracy']:.1%}")
    print(f"    Purity: {baseline['final_purity']['mean_purity']:.1%}")
    
    # Test 1: Shared dimensions
    print("\n" + "-" * 70)
    print("[TEST 1] Shared-Dimension Overlap")
    print("-" * 70)
    print("    Classes share dimensions: A∩B={2}, B∩C={4}, C∩D={6}, D∩E={8}")
    
    overlap_result = run_overlap_test()
    print(f"\n    Accuracy: {overlap_result['test_accuracy']:.1%}")
    print(f"    Mean Purity: {overlap_result['mean_purity']:.1%}")
    print("\n    Per-class purity:")
    for name, data in overlap_result["purity_per_class"].items():
        print(f"      {name}: tile={data['dominant_tile']}, purity={data['purity']:.1%}")
    
    print("\n    Learned signatures (positive dims):")
    for tile_idx, spec in overlap_result["specialization"].items():
        print(f"      Tile {tile_idx}: +{spec['pos']}, -{spec['neg']}")
    
    # Test 2: Gradient boundaries
    print("\n" + "-" * 70)
    print("[TEST 2] Gradient Boundaries (20% mixing)")
    print("-" * 70)
    
    gradient_result = run_gradient_test(mixing_prob=0.2)
    print(f"\n    Accuracy: {gradient_result['test_accuracy']:.1%}")
    print(f"    Mean Purity: {gradient_result['mean_purity']:.1%}")
    print("\n    Per-class purity:")
    for name, data in gradient_result["purity_per_class"].items():
        print(f"      {name}: tile={data['dominant_tile']}, purity={data['purity']:.1%}")
    
    # Test 2b: Higher mixing
    print("\n" + "-" * 70)
    print("[TEST 2b] Gradient Boundaries (40% mixing)")
    print("-" * 70)
    
    gradient_result_40 = run_gradient_test(mixing_prob=0.4)
    print(f"\n    Accuracy: {gradient_result_40['test_accuracy']:.1%}")
    print(f"    Mean Purity: {gradient_result_40['mean_purity']:.1%}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    baseline_purity = baseline['final_purity']['mean_purity']
    overlap_purity = overlap_result['mean_purity']
    gradient_purity = gradient_result['mean_purity']
    gradient_purity_40 = gradient_result_40['mean_purity']
    
    print(f"\n    Baseline purity (clean):        {baseline_purity:.1%}")
    print(f"    Overlap purity (shared dims):   {overlap_purity:.1%} ({overlap_purity - baseline_purity:+.1%})")
    print(f"    Gradient purity (20% mix):      {gradient_purity:.1%} ({gradient_purity - baseline_purity:+.1%})")
    print(f"    Gradient purity (40% mix):      {gradient_purity_40:.1%} ({gradient_purity_40 - baseline_purity:+.1%})")
    
    # Graceful degradation = purity drops but stays reasonable
    graceful = (
        overlap_purity > 0.7 and 
        gradient_purity > 0.7 and
        gradient_purity_40 > 0.5
    )
    
    collapse = overlap_purity < 0.5 or gradient_purity < 0.5
    
    if graceful:
        print("\n    ✓ GRACEFUL DEGRADATION CONFIRMED")
        print("    → Purity drops under fuzz but specialization persists.")
        print("    → Mechanism is robust to boundary ambiguity.")
    elif collapse:
        print("\n    ✗ COLLAPSE DETECTED")
        print("    → Mechanism breaks under fuzzy boundaries.")
    else:
        print("\n    ~ PARTIAL DEGRADATION")
        print("    → Some robustness, but needs investigation.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
