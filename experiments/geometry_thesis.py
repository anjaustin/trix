#!/usr/bin/env python3
"""
Minimal Geometry Thesis Experiment

Question: Does geometry alone suffice to express semantics as addressable computation?

Setup:
- 16-D input space with interpretable dimensions
- 5 semantic classes defined by geometric rules (not learned)
- Single TriX-style layer: ternary signatures, argmax routing
- Measure: signature stability, routing determinism, tile specialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# =============================================================================
# 1. Synthetic Semantic Space
# =============================================================================

@dataclass
class SemanticClass:
    """A semantic class defined by geometric rules."""
    name: str
    positive_dims: list  # dims that should be high
    negative_dims: list  # dims that should be low
    # Other dims are irrelevant (noise)


SEMANTIC_CLASSES = [
    # Fully orthogonal: each class owns non-overlapping dimension ranges
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
    """
    Generate synthetic data where semantics are geometric by construction.
    
    Returns:
        x: [n_samples, d_model] input vectors
        y: [n_samples] class labels (0 to 4)
    """
    n_classes = len(SEMANTIC_CLASSES)
    samples_per_class = n_samples // n_classes
    
    xs, ys = [], []
    
    for class_idx, sem_class in enumerate(SEMANTIC_CLASSES):
        # Start with noise
        x = torch.randn(samples_per_class, d_model) * noise_scale
        
        # Set positive dimensions high
        for dim in sem_class.positive_dims:
            x[:, dim] = signal_strength + torch.randn(samples_per_class) * noise_scale
        
        # Set negative dimensions low
        for dim in sem_class.negative_dims:
            x[:, dim] = -signal_strength + torch.randn(samples_per_class) * noise_scale
        
        xs.append(x)
        ys.append(torch.full((samples_per_class,), class_idx, dtype=torch.long))
    
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    
    # Shuffle
    perm = torch.randperm(len(x))
    return x[perm], y[perm]


# =============================================================================
# 2. Minimal TriX Layer (stripped to essence)
# =============================================================================

class MinimalTriXLayer(nn.Module):
    """
    Single TriX layer - the simplest thing that could work.
    
    - Ternary signatures per tile
    - Argmax routing (no softmax, no blending)
    - Each tile has its OWN output (routing must do all semantic work)
    """
    
    def __init__(self, d_model: int, n_tiles: int, n_classes: int):
        super().__init__()
        self.d_model = d_model
        self.n_tiles = n_tiles
        self.n_classes = n_classes
        
        # Learnable signatures (will be quantized to ternary)
        self.signatures_raw = nn.Parameter(torch.randn(n_tiles, d_model) * 0.5)
        
        # Each tile has its OWN output projection (no sharing!)
        self.tile_outputs = nn.ModuleList([
            nn.Linear(d_model, n_classes) for _ in range(n_tiles)
        ])
    
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
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, d_model]
        
        Returns:
            logits: [batch, n_classes]
            tile_indices: [batch] - which tile was selected
        """
        batch_size = x.shape[0]
        
        # Route: input · signature
        sigs = self.signatures  # [n_tiles, d_model]
        scores = x @ sigs.T     # [batch, n_tiles]
        
        # Argmax routing (hard selection)
        tile_indices = scores.argmax(dim=-1)  # [batch]
        
        # Each sample goes through its selected tile's output
        # (This forces routing to do the semantic separation)
        logits = torch.zeros(batch_size, self.n_classes, device=x.device)
        for tile_idx in range(self.n_tiles):
            mask = tile_indices == tile_idx
            if mask.any():
                logits[mask] = self.tile_outputs[tile_idx](x[mask])
        
        return logits, tile_indices
    
    def get_signature_analysis(self) -> dict:
        """Analyze what each tile's signature says about semantics."""
        sigs = self.signatures.detach()
        analysis = {}
        
        for tile_idx in range(self.n_tiles):
            sig = sigs[tile_idx]
            pos_dims = (sig > 0.5).nonzero(as_tuple=True)[0].tolist()
            neg_dims = (sig < -0.5).nonzero(as_tuple=True)[0].tolist()
            analysis[tile_idx] = {
                "positive_dims": pos_dims,
                "negative_dims": neg_dims,
                "signature": sig.tolist(),
            }
        
        return analysis
    
    def compute_routing_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Encourage inputs to route to their 'correct' tile.
        Uses soft routing scores to maintain gradient flow.
        """
        sigs = self.signatures_raw  # Use raw (continuous) for gradient flow
        scores = x @ sigs.T  # [batch, n_tiles]
        
        # Soft assignment: each class should maximize score to one tile
        log_probs = F.log_softmax(scores, dim=-1)
        
        # Simple approach: class i should route to tile i (when n_tiles == n_classes)
        if self.n_tiles == self.n_classes:
            target_tiles = y
            routing_loss = F.nll_loss(log_probs, target_tiles)
        else:
            routing_loss = torch.tensor(0.0, device=x.device)
        
        return routing_loss


# =============================================================================
# 3. Training & Measurement
# =============================================================================

def train_and_measure(
    n_epochs: int = 100,
    n_train: int = 1000,
    n_test: int = 500,
    d_model: int = 16,
    n_tiles: int = 5,  # Match n_classes for 1:1 mapping
    lr: float = 0.01,
    seed: int = 42,
):
    """Train minimal TriX and measure the three key metrics."""
    
    torch.manual_seed(seed)
    
    # Generate data
    x_train, y_train = generate_semantic_data(n_train, d_model)
    x_test, y_test = generate_semantic_data(n_test, d_model)
    
    # Model
    n_classes = len(SEMANTIC_CLASSES)
    model = MinimalTriXLayer(d_model, n_tiles, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("=" * 60)
    print("GEOMETRY THESIS EXPERIMENT")
    print("=" * 60)
    print(f"Config: d_model={d_model}, n_tiles={n_tiles}, n_classes={n_classes}")
    print(f"Training: {n_train} samples, {n_epochs} epochs")
    print()
    
    # Track signature evolution
    signature_history = []
    
    for epoch in range(n_epochs):
        model.train()
        
        logits, tile_indices = model(x_train)
        task_loss = F.cross_entropy(logits, y_train)
        routing_loss = model.compute_routing_loss(x_train, y_train)
        loss = task_loss + routing_loss  # Joint optimization
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record raw signatures every 10 epochs (to track learning)
        if epoch % 10 == 0:
            signature_history.append(model.signatures_raw.detach().clone())
        
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_logits, test_tiles = model(x_test)
                test_acc = (test_logits.argmax(-1) == y_test).float().mean()
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, test_acc={test_acc:.3f}")
    
    # ==========================================================================
    # MEASUREMENTS
    # ==========================================================================
    
    model.eval()
    print()
    print("=" * 60)
    print("MEASUREMENTS")
    print("=" * 60)
    
    # 1. Signature Stability
    print("\n[1] SIGNATURE STABILITY")
    if len(signature_history) >= 2:
        first_sigs = signature_history[0]
        last_sigs = signature_history[-1]
        
        # Measure how much raw signatures changed
        sig_change = (last_sigs - first_sigs).abs().mean().item()
        
        # Check if they converged to clean ternary (raw values near -1, 0, or +1)
        near_zero = (last_sigs.abs() < 0.2).float()
        near_one = (last_sigs.abs() > 0.8).float()
        ternary_clean = (near_zero + near_one).mean().item()
        
        print(f"    Mean signature change (first→last): {sig_change:.4f}")
        print(f"    Ternary cleanness (% near -1/0/+1): {ternary_clean:.1%}")
    
    # 2. Routing Determinism
    print("\n[2] ROUTING DETERMINISM")
    with torch.no_grad():
        _, tiles_1 = model(x_test)
        _, tiles_2 = model(x_test)
        determinism = (tiles_1 == tiles_2).float().mean().item()
        print(f"    Same input → same tile: {determinism:.1%}")
        
        # Check routing by class
        print("\n    Routing distribution by semantic class:")
        for class_idx, sem_class in enumerate(SEMANTIC_CLASSES):
            mask = y_test == class_idx
            class_tiles = tiles_1[mask]
            if len(class_tiles) > 0:
                dominant_tile = class_tiles.mode().values.item()
                purity = (class_tiles == dominant_tile).float().mean().item()
                print(f"      Class {sem_class.name}: dominant_tile={dominant_tile}, purity={purity:.1%}")
    
    # 3. Tile Specialization
    print("\n[3] TILE SPECIALIZATION")
    sig_analysis = model.get_signature_analysis()
    
    print("\n    Learned signatures vs ground truth:")
    for class_idx, sem_class in enumerate(SEMANTIC_CLASSES):
        print(f"\n    Class {sem_class.name} (truth: +dims={sem_class.positive_dims}, -dims={sem_class.negative_dims})")
        
        # Find which tile routes this class most
        mask = y_test == class_idx
        if mask.sum() > 0:
            with torch.no_grad():
                _, class_tiles = model(x_test[mask])
            dominant_tile = class_tiles.mode().values.item()
            
            tile_sig = sig_analysis[dominant_tile]
            print(f"      → Tile {dominant_tile}: +dims={tile_sig['positive_dims']}, -dims={tile_sig['negative_dims']}")
    
    # Final accuracy
    print()
    print("=" * 60)
    with torch.no_grad():
        final_logits, _ = model(x_test)
        final_acc = (final_logits.argmax(-1) == y_test).float().mean().item()
    print(f"FINAL TEST ACCURACY: {final_acc:.1%}")
    print("=" * 60)
    
    return model, signature_history


if __name__ == "__main__":
    model, history = train_and_measure()
