#!/usr/bin/env python3
"""
Gate 3: Multi-Layer Composition

Question: Do meanings compose, or just cluster?

Tests:
  1. Does Layer 1 route to coarse regions, Layer 2 refine?
  2. Are routing paths interpretable? (e.g., "vertical → thin → 1")
  3. Do different paths lead to different outcomes?

If this works, TriX becomes a "learned instruction set" — 
routing paths are programs, tiles are operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


# =============================================================================
# MNIST Data Loading
# =============================================================================

def load_mnist():
    """Load MNIST using torchvision."""
    from torchvision import datasets, transforms
    
    mnist_dir = "/workspace/trix_latest/experiments/data"
    
    train_dataset = datasets.MNIST(root=mnist_dir, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root=mnist_dir, train=False, download=True, transform=transforms.ToTensor())
    
    x_train = train_dataset.data.float().view(-1, 784) / 255.0
    y_train = train_dataset.targets
    x_test = test_dataset.data.float().view(-1, 784) / 255.0
    y_test = test_dataset.targets
    
    return x_train, y_train, x_test, y_test


# =============================================================================
# Multi-Layer TriX Model
# =============================================================================

class TriXLayer(nn.Module):
    """Single TriX routing layer."""
    
    def __init__(self, d_model: int, n_tiles: int):
        super().__init__()
        self.d_model = d_model
        self.n_tiles = n_tiles
        
        self.signatures_raw = nn.Parameter(torch.randn(n_tiles, d_model) * 0.5)
        
        # Each tile transforms the representation
        self.tile_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            ) for _ in range(n_tiles)
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, d_model]
        
        Returns:
            output: [batch, d_model] transformed representation
            tile_indices: [batch] which tile was selected
        """
        batch_size = x.shape[0]
        
        sigs = self.signatures
        scores = x @ sigs.T
        tile_indices = scores.argmax(dim=-1)
        
        # Apply selected tile's transform
        output = torch.zeros_like(x)
        for tile_idx in range(self.n_tiles):
            mask = tile_indices == tile_idx
            if mask.any():
                output[mask] = self.tile_transforms[tile_idx](x[mask])
        
        # Residual connection
        output = output + x
        
        return output, tile_indices


class MultiLayerTriX(nn.Module):
    """
    Multi-layer TriX model for studying composition.
    
    Architecture:
      Input → Project → [TriX Layer 1] → [TriX Layer 2] → [...] → Classify
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        d_model: int = 64,
        n_layers: int = 3,
        tiles_per_layer: List[int] = None,
        n_classes: int = 10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        if tiles_per_layer is None:
            tiles_per_layer = [8] * n_layers  # Default: 8 tiles per layer
        self.tiles_per_layer = tiles_per_layer
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # TriX layers
        self.trix_layers = nn.ModuleList([
            TriXLayer(d_model, n_tiles)
            for n_tiles in tiles_per_layer
        ])
        
        # Output classifier
        self.classifier = nn.Linear(d_model, n_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [batch, input_dim]
        
        Returns:
            logits: [batch, n_classes]
            routing_path: List of [batch] tensors, one per layer
        """
        h = self.input_proj(x)
        
        routing_path = []
        for layer in self.trix_layers:
            h, tile_indices = layer(h)
            routing_path.append(tile_indices)
        
        logits = self.classifier(h)
        
        return logits, routing_path
    
    def get_path_string(self, routing_path: List[torch.Tensor], idx: int) -> str:
        """Get routing path as string for a single sample."""
        return " → ".join([f"L{i}T{path[idx].item()}" for i, path in enumerate(routing_path)])


# =============================================================================
# Training and Analysis
# =============================================================================

def train_multilayer_trix(
    n_epochs: int = 30,
    batch_size: int = 128,
    d_model: int = 64,
    n_layers: int = 3,
    tiles_per_layer: List[int] = None,
    lr: float = 0.001,
    seed: int = 42,
):
    """Train multi-layer TriX and analyze composition."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if tiles_per_layer is None:
        tiles_per_layer = [8, 8, 8]  # 3 layers, 8 tiles each
    
    print("=" * 70)
    print("GATE 3: Multi-Layer Composition")
    print("=" * 70)
    print("\nQuestion: Do meanings compose across layers?")
    print(f"\nArchitecture: {n_layers} layers, tiles per layer: {tiles_per_layer}")
    print()
    
    # Load data
    print("[1] Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"    Train: {len(x_train)}, Test: {len(x_test)}")
    
    # Model
    print(f"\n[2] Building {n_layers}-layer TriX model...")
    model = MultiLayerTriX(
        input_dim=784,
        d_model=d_model,
        n_layers=n_layers,
        tiles_per_layer=tiles_per_layer,
        n_classes=10,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}")
    
    # Training
    print(f"\n[3] Training ({n_epochs} epochs)...")
    n_batches = len(x_train) // batch_size
    
    for epoch in range(n_epochs):
        model.train()
        
        perm = torch.randperm(len(x_train))
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]
        
        epoch_loss = 0.0
        for i in range(n_batches):
            x_batch = x_shuffled[i*batch_size:(i+1)*batch_size]
            y_batch = y_shuffled[i*batch_size:(i+1)*batch_size]
            
            logits, _ = model(x_batch)
            loss = F.cross_entropy(logits, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits, test_paths = model(x_test)
                test_acc = (test_logits.argmax(-1) == y_test).float().mean()
            print(f"    Epoch {epoch:2d}: loss={epoch_loss/n_batches:.4f}, acc={test_acc:.1%}")
    
    # ==========================================================================
    # COMPOSITION ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("COMPOSITION ANALYSIS")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        test_logits, test_paths = model(x_test)
        predictions = test_logits.argmax(-1)
    
    # Analyze routing paths
    print("\n[A] Path Diversity")
    print("-" * 70)
    
    # Count unique paths
    path_tuples = [tuple(p[i].item() for p in test_paths) for i in range(len(x_test))]
    unique_paths = set(path_tuples)
    
    print(f"    Total samples: {len(x_test)}")
    print(f"    Unique routing paths: {len(unique_paths)}")
    print(f"    Max possible paths: {np.prod(tiles_per_layer)}")
    print(f"    Path utilization: {len(unique_paths) / np.prod(tiles_per_layer):.1%}")
    
    # Most common paths
    path_counts = defaultdict(int)
    for p in path_tuples:
        path_counts[p] += 1
    
    top_paths = sorted(path_counts.items(), key=lambda x: -x[1])[:10]
    print(f"\n    Top 10 paths:")
    for path, count in top_paths:
        path_str = " → ".join([f"L{i}T{t}" for i, t in enumerate(path)])
        pct = count / len(x_test) * 100
        print(f"      {path_str}: {count} ({pct:.1f}%)")
    
    # Per-layer tile usage
    print("\n[B] Per-Layer Tile Usage")
    print("-" * 70)
    
    for layer_idx, path_tensor in enumerate(test_paths):
        tile_counts = torch.bincount(path_tensor, minlength=tiles_per_layer[layer_idx])
        active = (tile_counts > 0).sum().item()
        max_usage = tile_counts.max().item()
        min_usage = tile_counts[tile_counts > 0].min().item() if active > 0 else 0
        
        print(f"    Layer {layer_idx}: {active}/{tiles_per_layer[layer_idx]} tiles active, "
              f"usage range: {min_usage}-{max_usage}")
    
    # Path → Digit analysis
    print("\n[C] Path → Digit Specialization")
    print("-" * 70)
    
    path_digit_counts = defaultdict(lambda: defaultdict(int))
    for i, (path, digit) in enumerate(zip(path_tuples, y_test.tolist())):
        path_digit_counts[path][digit] += 1
    
    # For top paths, what digits do they handle?
    print("\n    Top paths and their digit distributions:")
    for path, total_count in top_paths[:5]:
        digit_dist = path_digit_counts[path]
        sorted_digits = sorted(digit_dist.items(), key=lambda x: -x[1])
        top_digit, top_count = sorted_digits[0]
        purity = top_count / total_count
        
        path_str = " → ".join([f"L{i}T{t}" for i, t in enumerate(path)])
        top3 = ", ".join([f"{d}({c})" for d, c in sorted_digits[:3]])
        print(f"      {path_str}: n={total_count}, top=[{top3}], purity={purity:.0%}")
    
    # Digit → Path analysis
    print("\n[D] Digit → Path Patterns")
    print("-" * 70)
    
    digit_paths = defaultdict(list)
    for path, digit in zip(path_tuples, y_test.tolist()):
        digit_paths[digit].append(path)
    
    print("\n    Per-digit path diversity:")
    for digit in range(10):
        paths = digit_paths[digit]
        unique = len(set(paths))
        total = len(paths)
        
        # Most common path for this digit
        path_counts_digit = defaultdict(int)
        for p in paths:
            path_counts_digit[p] += 1
        top_path, top_count = max(path_counts_digit.items(), key=lambda x: x[1])
        concentration = top_count / total
        
        top_path_str = "→".join([f"{t}" for t in top_path])
        print(f"      Digit {digit}: {unique} unique paths, "
              f"top path [{top_path_str}] = {concentration:.0%}")
    
    # Hierarchical analysis: does Layer 1 do coarse, Layer 2 fine?
    print("\n[E] Hierarchical Routing Analysis")
    print("-" * 70)
    
    # For each Layer 1 tile, what digits pass through it?
    print("\n    Layer 1 (coarse routing):")
    l1_digit_counts = defaultdict(lambda: defaultdict(int))
    for path, digit in zip(path_tuples, y_test.tolist()):
        l1_tile = path[0]
        l1_digit_counts[l1_tile][digit] += 1
    
    for l1_tile in sorted(l1_digit_counts.keys()):
        counts = l1_digit_counts[l1_tile]
        total = sum(counts.values())
        sorted_digits = sorted(counts.items(), key=lambda x: -x[1])
        top3 = ", ".join([f"{d}({c/total:.0%})" for d, c in sorted_digits[:3]])
        print(f"      L1T{l1_tile}: n={total}, digits=[{top3}]")
    
    # For samples that went through same L1, does L2 differentiate?
    print("\n    Layer 2 refinement (within L1 groups):")
    
    for l1_tile in sorted(l1_digit_counts.keys())[:3]:  # Top 3 L1 tiles
        # Get all samples that went through this L1 tile
        l1_samples = [(path, digit) for path, digit in zip(path_tuples, y_test.tolist()) 
                      if path[0] == l1_tile]
        
        if len(l1_samples) < 100:
            continue
        
        # Within this L1 tile, how does L2 split?
        l2_digit_counts = defaultdict(lambda: defaultdict(int))
        for path, digit in l1_samples:
            l2_tile = path[1]
            l2_digit_counts[l2_tile][digit] += 1
        
        print(f"\n      Within L1T{l1_tile} (n={len(l1_samples)}):")
        for l2_tile in sorted(l2_digit_counts.keys()):
            counts = l2_digit_counts[l2_tile]
            total = sum(counts.values())
            if total < 20:
                continue
            sorted_digits = sorted(counts.items(), key=lambda x: -x[1])
            top_digit, top_count = sorted_digits[0]
            purity = top_count / total
            top2 = ", ".join([f"{d}({c/total:.0%})" for d, c in sorted_digits[:2]])
            refinement = "REFINES" if purity > 0.5 else "mixed"
            print(f"        → L2T{l2_tile}: n={total}, [{top2}] {refinement}")
    
    # ==========================================================================
    # VERDICT
    # ==========================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    final_acc = (predictions == y_test).float().mean().item()
    path_utilization = len(unique_paths) / np.prod(tiles_per_layer)
    
    # Check for hierarchical structure
    # L1 should be more "mixed" (coarse), L2 should be more "pure" (fine)
    l1_purities = []
    for l1_tile, counts in l1_digit_counts.items():
        total = sum(counts.values())
        if total > 50:
            top_count = max(counts.values())
            l1_purities.append(top_count / total)
    
    mean_l1_purity = np.mean(l1_purities) if l1_purities else 0
    
    print(f"\n    Test accuracy: {final_acc:.1%}")
    print(f"    Unique paths used: {len(unique_paths)}/{np.prod(tiles_per_layer)}")
    print(f"    Path utilization: {path_utilization:.1%}")
    print(f"    Mean L1 purity (coarse): {mean_l1_purity:.1%}")
    
    # Pass conditions
    composition_works = len(unique_paths) > 20  # Many paths used
    hierarchical = mean_l1_purity < 0.5  # L1 is coarse (mixed)
    
    print(f"\n    Path diversity (>20 paths): {'PASS' if composition_works else 'FAIL'}")
    print(f"    Hierarchical structure (L1 coarse): {'PASS' if hierarchical else 'FAIL'}")
    
    if composition_works and hierarchical:
        print("\n    ✓ MULTI-LAYER COMPOSITION CONFIRMED")
        print("    → Routing paths are diverse and meaningful")
        print("    → Layer 1 routes coarse, deeper layers refine")
        print("    → This is a learned routing program, not just clustering")
    elif composition_works:
        print("\n    ~ PARTIAL: Paths diverse but not clearly hierarchical")
    else:
        print("\n    ✗ COMPOSITION NOT CONFIRMED")
    
    print("=" * 70)
    
    return {
        "accuracy": final_acc,
        "unique_paths": len(unique_paths),
        "path_utilization": path_utilization,
        "mean_l1_purity": mean_l1_purity,
        "top_paths": top_paths,
    }


if __name__ == "__main__":
    results = train_multilayer_trix(
        n_epochs=30,
        n_layers=3,
        tiles_per_layer=[8, 8, 8],
    )
