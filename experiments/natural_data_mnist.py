#!/usr/bin/env python3
"""
Gate 2: Natural Data Pilot — MNIST

Question: Do interpretable islands emerge in real data?

We're NOT chasing SOTA. We're testing:
  1. Do tiles specialize by digit class?
  2. Are the specializations interpretable?
  3. Does routing remain stable and deterministic?

Pass condition: Tiles show clear digit preferences that we can name.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


# =============================================================================
# MNIST Data Loading (using torchvision)
# =============================================================================

def load_mnist():
    """Load MNIST using torchvision."""
    from torchvision import datasets, transforms
    
    mnist_dir = "/workspace/trix_latest/experiments/data"
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root=mnist_dir, 
        train=True, 
        download=True,
        transform=transforms.ToTensor()
    )
    test_dataset = datasets.MNIST(
        root=mnist_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Convert to tensors
    x_train = train_dataset.data.float().view(-1, 784) / 255.0
    y_train = train_dataset.targets
    x_test = test_dataset.data.float().view(-1, 784) / 255.0
    y_test = test_dataset.targets
    
    return x_train, y_train, x_test, y_test


# =============================================================================
# Model: TriX for MNIST
# =============================================================================

class MNISTTriXLayer(nn.Module):
    """
    TriX layer for MNIST classification.
    
    Architecture:
      Input (784) → Project to d_model → TriX routing → Classification
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        d_model: int = 64,
        n_tiles: int = 16,
        n_classes: int = 10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_tiles = n_tiles
        self.n_classes = n_classes
        
        # Project input to routing space
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Ternary signatures for routing
        self.signatures_raw = nn.Parameter(torch.randn(n_tiles, d_model) * 0.5)
        
        # Each tile has its own classifier
        self.tile_classifiers = nn.ModuleList([
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
        """
        Args:
            x: [batch, 784] flattened MNIST images
        
        Returns:
            logits: [batch, 10] class predictions
            tile_indices: [batch] which tile was selected
        """
        batch_size = x.shape[0]
        
        # Project to routing space
        h = self.input_proj(x)  # [batch, d_model]
        
        # Route
        sigs = self.signatures
        scores = h @ sigs.T  # [batch, n_tiles]
        tile_indices = scores.argmax(dim=-1)
        
        # Each sample goes through its tile's classifier
        logits = torch.zeros(batch_size, self.n_classes, device=x.device)
        for tile_idx in range(self.n_tiles):
            mask = tile_indices == tile_idx
            if mask.any():
                logits[mask] = self.tile_classifiers[tile_idx](h[mask])
        
        return logits, tile_indices
    
    def compute_routing_entropy_loss(self, tile_indices: torch.Tensor) -> torch.Tensor:
        """Encourage balanced tile usage."""
        counts = torch.bincount(tile_indices, minlength=self.n_tiles).float()
        probs = counts / counts.sum()
        entropy = -(probs * (probs + 1e-10).log()).sum()
        max_entropy = np.log(self.n_tiles)
        return -entropy / max_entropy  # Negative because we want to maximize entropy


# =============================================================================
# Training and Analysis
# =============================================================================

def train_mnist_trix(
    n_epochs: int = 20,
    batch_size: int = 128,
    d_model: int = 64,
    n_tiles: int = 16,
    lr: float = 0.001,
    seed: int = 42,
):
    """Train TriX on MNIST and analyze tile specialization."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("GATE 2: Natural Data Pilot — MNIST")
    print("=" * 70)
    print("\nQuestion: Do interpretable islands emerge in real data?")
    print()
    
    # Load data
    print("[1] Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"    Train: {len(x_train)} samples")
    print(f"    Test: {len(x_test)} samples")
    
    # Model
    print(f"\n[2] Model setup: d_model={d_model}, n_tiles={n_tiles}")
    model = MNISTTriXLayer(
        input_dim=784,
        d_model=d_model,
        n_tiles=n_tiles,
        n_classes=10,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    print(f"\n[3] Training ({n_epochs} epochs)...")
    n_batches = len(x_train) // batch_size
    
    for epoch in range(n_epochs):
        model.train()
        
        # Shuffle
        perm = torch.randperm(len(x_train))
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm]
        
        epoch_loss = 0.0
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            
            x_batch = x_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            
            logits, tile_indices = model(x_batch)
            task_loss = F.cross_entropy(logits, y_batch)
            entropy_loss = model.compute_routing_entropy_loss(tile_indices)
            
            loss = task_loss + 0.1 * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += task_loss.item()
        
        # Evaluate
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits, test_tiles = model(x_test)
                test_acc = (test_logits.argmax(-1) == y_test).float().mean()
                
                # Tile usage
                tile_counts = torch.bincount(test_tiles, minlength=n_tiles)
                active_tiles = (tile_counts > 0).sum().item()
            
            print(f"    Epoch {epoch:2d}: loss={epoch_loss/n_batches:.4f}, "
                  f"acc={test_acc:.1%}, active_tiles={active_tiles}/{n_tiles}")
    
    # ==========================================================================
    # TILE SPECIALIZATION ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TILE SPECIALIZATION ANALYSIS")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        _, test_tiles = model(x_test)
    
    # For each tile, what digits does it prefer?
    print("\n[A] Tile → Digit Preferences")
    print("-" * 70)
    
    tile_digit_counts = defaultdict(lambda: defaultdict(int))
    for tile_idx, digit in zip(test_tiles.tolist(), y_test.tolist()):
        tile_digit_counts[tile_idx][digit] += 1
    
    tile_specializations = {}
    for tile_idx in range(n_tiles):
        counts = tile_digit_counts[tile_idx]
        total = sum(counts.values())
        
        if total == 0:
            print(f"    Tile {tile_idx:2d}: UNUSED")
            continue
        
        # Sort by count
        sorted_digits = sorted(counts.items(), key=lambda x: -x[1])
        top_digit, top_count = sorted_digits[0]
        purity = top_count / total
        
        # Top 3 digits
        top3 = [(d, c/total) for d, c in sorted_digits[:3]]
        top3_str = ", ".join([f"{d}({p:.0%})" for d, p in top3])
        
        tile_specializations[tile_idx] = {
            "top_digit": top_digit,
            "purity": purity,
            "total": total,
            "distribution": dict(counts),
        }
        
        specialization = "STRONG" if purity > 0.5 else "MIXED" if purity > 0.3 else "WEAK"
        print(f"    Tile {tile_idx:2d}: {top3_str} [{specialization}] (n={total})")
    
    # For each digit, which tile handles it?
    print("\n[B] Digit → Tile Routing")
    print("-" * 70)
    
    digit_tile_counts = defaultdict(lambda: defaultdict(int))
    for tile_idx, digit in zip(test_tiles.tolist(), y_test.tolist()):
        digit_tile_counts[digit][tile_idx] += 1
    
    digit_routing = {}
    for digit in range(10):
        counts = digit_tile_counts[digit]
        total = sum(counts.values())
        
        sorted_tiles = sorted(counts.items(), key=lambda x: -x[1])
        top_tile, top_count = sorted_tiles[0]
        purity = top_count / total
        
        digit_routing[digit] = {
            "dominant_tile": top_tile,
            "purity": purity,
        }
        
        top3 = [(t, c/total) for t, c in sorted_tiles[:3]]
        top3_str = ", ".join([f"T{t}({p:.0%})" for t, p in top3])
        
        print(f"    Digit {digit}: {top3_str}")
    
    # Routing determinism
    print("\n[C] Routing Consistency")
    print("-" * 70)
    
    # Run twice, check if same routing
    with torch.no_grad():
        _, tiles1 = model(x_test)
        _, tiles2 = model(x_test)
    
    determinism = (tiles1 == tiles2).float().mean().item()
    print(f"    Same input → same tile: {determinism:.1%}")
    
    # Per-digit purity
    mean_digit_purity = np.mean([d["purity"] for d in digit_routing.values()])
    print(f"    Mean digit routing purity: {mean_digit_purity:.1%}")
    
    # ==========================================================================
    # VERDICT
    # ==========================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    final_acc = (model(x_test)[0].argmax(-1) == y_test).float().mean().item()
    
    # Check for interpretable specialization
    strong_specializations = sum(1 for s in tile_specializations.values() if s["purity"] > 0.5)
    
    print(f"\n    Test accuracy: {final_acc:.1%}")
    print(f"    Active tiles: {len(tile_specializations)}/{n_tiles}")
    print(f"    Strongly specialized tiles (>50% one digit): {strong_specializations}")
    print(f"    Mean digit→tile purity: {mean_digit_purity:.1%}")
    print(f"    Routing determinism: {determinism:.1%}")
    
    # Pass conditions
    specialization_emerged = strong_specializations >= 3
    routing_stable = determinism > 0.99
    reasonable_accuracy = final_acc > 0.90
    
    print(f"\n    Specialization emerged: {'PASS' if specialization_emerged else 'FAIL'}")
    print(f"    Routing stable: {'PASS' if routing_stable else 'FAIL'}")
    print(f"    Reasonable accuracy: {'PASS' if reasonable_accuracy else 'FAIL'}")
    
    if specialization_emerged and routing_stable:
        print("\n    ✓ NATURAL DATA PILOT SUCCESSFUL")
        print("    → Tiles specialize by digit without explicit supervision")
        print("    → Routing is deterministic and stable")
        print("    → Islands emerge in real data")
    else:
        print("\n    ~ PARTIAL SUCCESS or FAILURE")
        print("    → Review tile specializations for interpretability")
    
    print("=" * 70)
    
    return {
        "accuracy": final_acc,
        "tile_specializations": tile_specializations,
        "digit_routing": digit_routing,
        "determinism": determinism,
        "mean_purity": mean_digit_purity,
    }


if __name__ == "__main__":
    results = train_mnist_trix()
