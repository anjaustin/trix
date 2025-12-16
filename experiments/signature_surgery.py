#!/usr/bin/env python3
"""
Signature Surgery: Can we NAME meaning explicitly?

Test:
  1. Insert a hand-designed signature for Class A (dims 0,1,2)
  2. Freeze it during initial training
  3. Check if it claims Class A inputs
  4. Unfreeze and train more
  5. Check if it drifts or stays stable

Pass condition: Hand-designed signature claims its region and resists drift.
Implication: Explicit semantic control is possible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


# =============================================================================
# Data Generation (from convergence_test.py)
# =============================================================================

@dataclass
class SemanticClass:
    name: str
    positive_dims: list
    negative_dims: list


SEMANTIC_CLASSES = [
    SemanticClass("A", positive_dims=[0, 1, 2], negative_dims=[]),
    SemanticClass("B", positive_dims=[3, 4, 5], negative_dims=[]),
    SemanticClass("C", positive_dims=[6, 7, 8], negative_dims=[]),
    SemanticClass("D", positive_dims=[9, 10, 11], negative_dims=[]),
    SemanticClass("E", positive_dims=[12, 13, 14], negative_dims=[]),
]


def generate_semantic_data(
    n_samples: int,
    d_model: int = 16,
    signal_strength: float = 1.0,
    noise_scale: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_classes = len(SEMANTIC_CLASSES)
    samples_per_class = n_samples // n_classes
    
    xs, ys = [], []
    for class_idx, sem_class in enumerate(SEMANTIC_CLASSES):
        x = torch.randn(samples_per_class, d_model) * noise_scale
        for dim in sem_class.positive_dims:
            x[:, dim] = signal_strength + torch.randn(samples_per_class) * noise_scale
        for dim in sem_class.negative_dims:
            x[:, dim] = -signal_strength + torch.randn(samples_per_class) * noise_scale
        xs.append(x)
        ys.append(torch.full((samples_per_class,), class_idx, dtype=torch.long))
    
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    perm = torch.randperm(len(x))
    return x[perm], y[perm]


# =============================================================================
# Model with Surgical Signature Control
# =============================================================================

class SurgicalTriXLayer(nn.Module):
    """
    TriX layer with ability to freeze individual tile signatures.
    """
    
    def __init__(self, d_model: int, n_tiles: int, n_classes: int):
        super().__init__()
        self.d_model = d_model
        self.n_tiles = n_tiles
        self.n_classes = n_classes
        
        # Each signature is a separate parameter for fine-grained control
        self.signatures_raw = nn.ParameterList([
            nn.Parameter(torch.randn(d_model) * 0.5)
            for _ in range(n_tiles)
        ])
        
        self.tile_outputs = nn.ModuleList([
            nn.Linear(d_model, n_classes) for _ in range(n_tiles)
        ])
        
        # Track which signatures are frozen
        self.frozen_tiles = set()
    
    def freeze_signature(self, tile_idx: int):
        """Freeze a tile's signature (no gradient updates)."""
        self.signatures_raw[tile_idx].requires_grad = False
        self.frozen_tiles.add(tile_idx)
    
    def unfreeze_signature(self, tile_idx: int):
        """Unfreeze a tile's signature."""
        self.signatures_raw[tile_idx].requires_grad = True
        self.frozen_tiles.discard(tile_idx)
    
    def set_signature(self, tile_idx: int, signature: torch.Tensor):
        """Manually set a tile's signature."""
        with torch.no_grad():
            self.signatures_raw[tile_idx].copy_(signature)
    
    def _quantize_ternary(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = torch.zeros_like(x)
            q[x > 0.3] = 1.0
            q[x < -0.3] = -1.0
        return x + (q - x).detach()
    
    @property
    def signatures(self) -> torch.Tensor:
        """Get all signatures as a single tensor."""
        raw = torch.stack([s for s in self.signatures_raw])
        return self._quantize_ternary(raw)
    
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
        raw = torch.stack([s for s in self.signatures_raw])
        scores = x @ raw.T
        log_probs = F.log_softmax(scores, dim=-1)
        if self.n_tiles == self.n_classes:
            return F.nll_loss(log_probs, y)
        return torch.tensor(0.0, device=x.device)


# =============================================================================
# Surgery Experiment
# =============================================================================

def run_signature_surgery(
    seed: int = 42,
    d_model: int = 16,
    n_train: int = 1000,
    n_test: int = 500,
    frozen_epochs: int = 50,
    unfrozen_epochs: int = 50,
    lr: float = 0.01,
):
    """
    The surgery experiment:
    1. Create model with random signatures
    2. Surgically insert ground-truth signature for Class A into Tile 0
    3. Freeze Tile 0's signature
    4. Train for frozen_epochs
    5. Check if Tile 0 claims Class A
    6. Unfreeze Tile 0
    7. Train for unfrozen_epochs more
    8. Check if signature drifted or stayed stable
    """
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("SIGNATURE SURGERY: Can we NAME meaning explicitly?")
    print("=" * 70)
    
    # Setup
    n_classes = len(SEMANTIC_CLASSES)
    n_tiles = n_classes
    
    x_train, y_train = generate_semantic_data(n_train, d_model)
    x_test, y_test = generate_semantic_data(n_test, d_model)
    
    model = SurgicalTriXLayer(d_model, n_tiles, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # ==========================================================================
    # STEP 1: Surgical Insertion
    # ==========================================================================
    print("\n[STEP 1] Surgical Insertion")
    print("-" * 70)
    
    # Create ground-truth signature for Class A
    class_a = SEMANTIC_CLASSES[0]
    ground_truth_sig = torch.zeros(d_model)
    for dim in class_a.positive_dims:
        ground_truth_sig[dim] = 1.0  # Will quantize to +1
    for dim in class_a.negative_dims:
        ground_truth_sig[dim] = -1.0  # Will quantize to -1
    
    print(f"    Target: Class A (positive dims: {class_a.positive_dims})")
    print(f"    Designed signature: +1 on dims {class_a.positive_dims}, 0 elsewhere")
    
    # Insert into Tile 0
    model.set_signature(0, ground_truth_sig)
    model.freeze_signature(0)
    
    initial_sig = model.signatures[0].detach().clone()
    print(f"    Inserted into Tile 0 (frozen)")
    print(f"    Initial Tile 0 signature: {(initial_sig > 0.5).nonzero(as_tuple=True)[0].tolist()}")
    
    # ==========================================================================
    # STEP 2: Frozen Training
    # ==========================================================================
    print(f"\n[STEP 2] Frozen Training ({frozen_epochs} epochs)")
    print("-" * 70)
    
    for epoch in range(frozen_epochs):
        model.train()
        logits, _ = model(x_train)
        task_loss = F.cross_entropy(logits, y_train)
        routing_loss = model.compute_routing_loss(x_train, y_train)
        loss = task_loss + routing_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == frozen_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits, test_tiles = model(x_test)
                test_acc = (test_logits.argmax(-1) == y_test).float().mean()
                
                # Check Class A routing to Tile 0
                class_a_mask = y_test == 0
                class_a_tiles = test_tiles[class_a_mask]
                tile_0_claim = (class_a_tiles == 0).float().mean()
            
            print(f"    Epoch {epoch:3d}: acc={test_acc:.1%}, "
                  f"Tile 0 claims Class A: {tile_0_claim:.1%}")
    
    # Check frozen signature didn't change
    frozen_sig = model.signatures[0].detach().clone()
    sig_changed = not torch.allclose(initial_sig, frozen_sig)
    print(f"\n    Signature changed during frozen training: {sig_changed}")
    
    # Detailed routing analysis
    model.eval()
    with torch.no_grad():
        _, test_tiles = model(x_test)
    
    print("\n    Routing after frozen training:")
    for class_idx, sem_class in enumerate(SEMANTIC_CLASSES):
        mask = y_test == class_idx
        class_tiles = test_tiles[mask]
        dominant = class_tiles.mode().values.item()
        purity = (class_tiles == dominant).float().mean().item()
        tile_0_frac = (class_tiles == 0).float().mean().item()
        print(f"      Class {sem_class.name}: dominant=Tile {dominant} ({purity:.0%}), "
              f"→Tile 0: {tile_0_frac:.0%}")
    
    # ==========================================================================
    # STEP 3: Unfreeze and Continue Training
    # ==========================================================================
    print(f"\n[STEP 3] Unfrozen Training ({unfrozen_epochs} epochs)")
    print("-" * 70)
    
    model.unfreeze_signature(0)
    pre_unfreeze_sig = model.signatures[0].detach().clone()
    
    print(f"    Tile 0 unfrozen. Watching for drift...")
    
    for epoch in range(unfrozen_epochs):
        model.train()
        logits, _ = model(x_train)
        task_loss = F.cross_entropy(logits, y_train)
        routing_loss = model.compute_routing_loss(x_train, y_train)
        loss = task_loss + routing_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == unfrozen_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits, test_tiles = model(x_test)
                test_acc = (test_logits.argmax(-1) == y_test).float().mean()
                
                class_a_mask = y_test == 0
                class_a_tiles = test_tiles[class_a_mask]
                tile_0_claim = (class_a_tiles == 0).float().mean()
                
                # Check drift
                current_sig = model.signatures[0]
                drift = (current_sig - pre_unfreeze_sig).abs().mean().item()
            
            print(f"    Epoch {epoch:3d}: acc={test_acc:.1%}, "
                  f"Tile 0 claims Class A: {tile_0_claim:.1%}, drift={drift:.4f}")
    
    # ==========================================================================
    # FINAL ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)
    
    final_sig = model.signatures[0].detach()
    final_pos_dims = (final_sig > 0.5).nonzero(as_tuple=True)[0].tolist()
    final_neg_dims = (final_sig < -0.5).nonzero(as_tuple=True)[0].tolist()
    
    print(f"\n    Designed signature: +dims {class_a.positive_dims}")
    print(f"    Final signature:    +dims {final_pos_dims}, -dims {final_neg_dims}")
    
    # Did it keep the core?
    designed_dims = set(class_a.positive_dims)
    final_dims = set(final_pos_dims)
    core_retained = designed_dims <= final_dims
    
    print(f"\n    Core semantic dims retained: {core_retained}")
    if core_retained:
        added_dims = final_dims - designed_dims
        if added_dims:
            print(f"    Additional dims learned: {sorted(added_dims)}")
    
    # Final routing
    model.eval()
    with torch.no_grad():
        _, test_tiles = model(x_test)
    
    class_a_mask = y_test == 0
    final_claim = (test_tiles[class_a_mask] == 0).float().mean().item()
    
    print(f"\n    Final Tile 0 claim on Class A: {final_claim:.1%}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    claim_success = final_claim > 0.9
    stability_success = core_retained
    
    print(f"\n    Hand-designed signature claimed target class: "
          f"{'PASS' if claim_success else 'FAIL'} ({final_claim:.1%})")
    print(f"    Signature stable after unfreeze: "
          f"{'PASS' if stability_success else 'FAIL'} (core dims retained: {core_retained})")
    
    if claim_success and stability_success:
        print("\n    ✓ SIGNATURE SURGERY SUCCESSFUL")
        print("    → We can NAME meaning explicitly")
        print("    → Hand-designed addresses claim their semantic region")
        print("    → Signatures resist drift toward other configurations")
        print("\n    IMPLICATION: Explicit semantic control is possible.")
    elif claim_success:
        print("\n    ~ PARTIAL SUCCESS")
        print("    → Claiming works, but signature drifted")
    else:
        print("\n    ✗ SURGERY FAILED")
        print("    → Hand-designed signature did not claim target region")
    
    print("=" * 70)
    
    return {
        "claim_rate": final_claim,
        "core_retained": core_retained,
        "final_signature": final_sig.tolist(),
        "designed_dims": class_a.positive_dims,
        "final_pos_dims": final_pos_dims,
        "final_neg_dims": final_neg_dims,
    }


if __name__ == "__main__":
    result = run_signature_surgery()
