#!/usr/bin/env python3
"""
Convergence Test: Do signatures recover structure across seeds?

Mission: Prove this is reproducible structure, not single-run coincidence.

Metrics:
  A) Permutation-invariant signature alignment (Hungarian matching)
  B) Routing purity distribution over training
  C) Semantic dims recovered score

Pass condition: Consistent structure across 10 seeds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from dataclasses import dataclass
from typing import Optional
import json


# =============================================================================
# Synthetic Semantic Space (same as geometry_thesis.py)
# =============================================================================

@dataclass
class SemanticClass:
    name: str
    positive_dims: list
    negative_dims: list


# Fully orthogonal classes for clean test
SEMANTIC_CLASSES = [
    SemanticClass("A", positive_dims=[0, 1, 2], negative_dims=[]),
    SemanticClass("B", positive_dims=[3, 4, 5], negative_dims=[]),
    SemanticClass("C", positive_dims=[6, 7, 8], negative_dims=[]),
    SemanticClass("D", positive_dims=[9, 10, 11], negative_dims=[]),
    SemanticClass("E", positive_dims=[12, 13, 14], negative_dims=[]),
]


def get_ground_truth_signatures(d_model: int = 16) -> torch.Tensor:
    """
    Build the 'ideal' ternary signatures for each class.
    These are what we expect the model to discover.
    """
    n_classes = len(SEMANTIC_CLASSES)
    signatures = torch.zeros(n_classes, d_model)
    
    for i, sem_class in enumerate(SEMANTIC_CLASSES):
        for dim in sem_class.positive_dims:
            signatures[i, dim] = 1.0
        for dim in sem_class.negative_dims:
            signatures[i, dim] = -1.0
    
    return signatures


def generate_semantic_data(
    n_samples: int,
    d_model: int = 16,
    signal_strength: float = 1.0,
    noise_scale: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data with geometric semantics."""
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
# Model (same as geometry_thesis.py but with tracking)
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
# Metrics
# =============================================================================

def hungarian_signature_alignment(
    learned_sigs: torch.Tensor,
    true_sigs: torch.Tensor,
) -> dict:
    """
    Compute permutation-invariant alignment between learned and true signatures.
    Uses Hungarian algorithm for optimal assignment.
    
    Returns:
        - assignment: mapping from learned tile -> true class
        - similarities: cosine similarity scores
        - containment: do learned sigs CONTAIN the true semantic dims?
    """
    learned = learned_sigs.detach().cpu().numpy()
    true = true_sigs.detach().cpu().numpy()
    
    # Compute similarity matrix (cosine similarity)
    learned_norm = learned / (np.linalg.norm(learned, axis=1, keepdims=True) + 1e-8)
    true_norm = true / (np.linalg.norm(true, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = learned_norm @ true_norm.T
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
    similarities = similarity_matrix[row_ind, col_ind]
    
    # NEW: Containment metric - do learned signatures contain true semantic dims?
    # This is more relevant than cosine similarity when signatures add discriminative negatives
    containment_scores = []
    for tile_idx, class_idx in zip(row_ind, col_ind):
        true_pos_dims = set(np.where(true[class_idx] > 0.5)[0])
        learned_pos_dims = set(np.where(learned[tile_idx] > 0.5)[0])
        
        if len(true_pos_dims) > 0:
            # What fraction of true dims are in learned?
            containment = len(true_pos_dims & learned_pos_dims) / len(true_pos_dims)
        else:
            containment = 1.0
        containment_scores.append(containment)
    
    return {
        "assignment": list(zip(row_ind.tolist(), col_ind.tolist())),
        "similarities": similarities.tolist(),
        "mean_similarity": float(similarities.mean()),
        "containment_scores": containment_scores,
        "mean_containment": float(np.mean(containment_scores)),
        "perfect_containment": int(sum(c == 1.0 for c in containment_scores)),
    }


def compute_routing_purity(
    model: MinimalTriXLayer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict:
    """Compute routing purity per class."""
    model.eval()
    with torch.no_grad():
        _, tile_indices = model(x)
    
    purity_per_class = {}
    for class_idx, sem_class in enumerate(SEMANTIC_CLASSES):
        mask = y == class_idx
        if mask.sum() > 0:
            class_tiles = tile_indices[mask]
            if len(class_tiles) > 0:
                dominant = class_tiles.mode().values.item()
                purity = (class_tiles == dominant).float().mean().item()
                purity_per_class[sem_class.name] = {
                    "dominant_tile": dominant,
                    "purity": purity,
                }
    
    mean_purity = np.mean([v["purity"] for v in purity_per_class.values()])
    return {
        "per_class": purity_per_class,
        "mean_purity": float(mean_purity),
    }


def compute_semantic_dims_recovered(
    learned_sigs: torch.Tensor,
    assignment: list[tuple[int, int]],
) -> dict:
    """
    For each tile, measure overlap with true semantic dims.
    Separately track 'discriminative negatives'.
    """
    learned = learned_sigs.detach()
    results = {}
    
    for tile_idx, class_idx in assignment:
        sem_class = SEMANTIC_CLASSES[class_idx]
        sig = learned[tile_idx]
        
        # True positive dims
        true_pos = set(sem_class.positive_dims)
        learned_pos = set((sig > 0.5).nonzero(as_tuple=True)[0].tolist())
        
        # True negative dims  
        true_neg = set(sem_class.negative_dims)
        learned_neg = set((sig < -0.5).nonzero(as_tuple=True)[0].tolist())
        
        # Compute overlaps
        pos_recovered = len(true_pos & learned_pos)
        pos_total = len(true_pos)
        
        # Discriminative negatives (learned negatives not in true negatives)
        # These are useful for discrimination, not errors
        extra_negatives = learned_neg - true_neg
        
        results[sem_class.name] = {
            "tile_idx": tile_idx,
            "pos_recovered": pos_recovered,
            "pos_total": pos_total,
            "pos_recovery_rate": pos_recovered / pos_total if pos_total > 0 else 1.0,
            "learned_positive_dims": sorted(learned_pos),
            "learned_negative_dims": sorted(learned_neg),
            "discriminative_negatives": sorted(extra_negatives),
        }
    
    mean_recovery = np.mean([v["pos_recovery_rate"] for v in results.values()])
    return {
        "per_class": results,
        "mean_pos_recovery": float(mean_recovery),
    }


# =============================================================================
# Training with Tracking
# =============================================================================

def train_single_seed(
    seed: int,
    n_epochs: int = 100,
    n_train: int = 1000,
    n_test: int = 500,
    d_model: int = 16,
    track_every: int = 10,
) -> dict:
    """Train one model, return all metrics."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_classes = len(SEMANTIC_CLASSES)
    n_tiles = n_classes  # 1:1 mapping
    
    x_train, y_train = generate_semantic_data(n_train, d_model)
    x_test, y_test = generate_semantic_data(n_test, d_model)
    true_sigs = get_ground_truth_signatures(d_model)
    
    model = MinimalTriXLayer(d_model, n_tiles, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Tracking
    purity_history = []
    alignment_history = []
    
    for epoch in range(n_epochs):
        model.train()
        logits, _ = model(x_train)
        task_loss = F.cross_entropy(logits, y_train)
        routing_loss = model.compute_routing_loss(x_train, y_train)
        loss = task_loss + routing_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics periodically
        if epoch % track_every == 0 or epoch == n_epochs - 1:
            purity = compute_routing_purity(model, x_test, y_test)
            alignment = hungarian_signature_alignment(model.signatures, true_sigs)
            
            purity_history.append({
                "epoch": epoch,
                "mean_purity": purity["mean_purity"],
            })
            alignment_history.append({
                "epoch": epoch,
                "mean_similarity": alignment["mean_similarity"],
            })
    
    # Final metrics
    model.eval()
    final_alignment = hungarian_signature_alignment(model.signatures, true_sigs)
    final_purity = compute_routing_purity(model, x_test, y_test)
    final_dims = compute_semantic_dims_recovered(
        model.signatures, 
        final_alignment["assignment"]
    )
    
    with torch.no_grad():
        test_logits, _ = model(x_test)
        test_acc = (test_logits.argmax(-1) == y_test).float().mean().item()
    
    return {
        "seed": seed,
        "test_accuracy": test_acc,
        "final_alignment": final_alignment,
        "final_purity": final_purity,
        "final_dims_recovered": final_dims,
        "purity_history": purity_history,
        "alignment_history": alignment_history,
        "final_signatures": model.signatures.detach().tolist(),
    }


# =============================================================================
# Main: Run Convergence Test
# =============================================================================

def run_convergence_test(n_seeds: int = 10):
    """Run multiple seeds and analyze convergence."""
    
    print("=" * 70)
    print("CONVERGENCE TEST: Do signatures recover structure across seeds?")
    print("=" * 70)
    print(f"\nRunning {n_seeds} seeds...\n")
    
    results = []
    for seed in range(n_seeds):
        print(f"  Seed {seed}...", end=" ", flush=True)
        result = train_single_seed(seed)
        results.append(result)
        print(f"acc={result['test_accuracy']:.1%}, "
              f"purity={result['final_purity']['mean_purity']:.1%}, "
              f"alignment={result['final_alignment']['mean_similarity']:.3f}")
    
    # Aggregate analysis
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    
    # A) Signature Alignment
    print("\n[A] SIGNATURE ALIGNMENT (Hungarian matching)")
    similarities = [r["final_alignment"]["mean_similarity"] for r in results]
    containments = [r["final_alignment"]["mean_containment"] for r in results]
    perfect_containments = [r["final_alignment"]["perfect_containment"] for r in results]
    
    print(f"    Cosine similarity: {np.mean(similarities):.3f} ± {np.std(similarities):.3f}")
    print(f"    Containment (semantic dims captured): {np.mean(containments):.1%} ± {np.std(containments):.1%}")
    print(f"    Perfect containment (all dims): {np.mean(perfect_containments):.1f} ± {np.std(perfect_containments):.1f} / 5")
    
    # B) Routing Purity
    print("\n[B] ROUTING PURITY")
    purities = [r["final_purity"]["mean_purity"] for r in results]
    print(f"    Mean purity: {np.mean(purities):.1%} ± {np.std(purities):.1%}")
    
    # Check learning trajectory consistency
    print("\n    Learning trajectory (purity over epochs):")
    for seed_idx in [0, n_seeds//2, n_seeds-1]:
        history = results[seed_idx]["purity_history"]
        trajectory = [f"{h['mean_purity']:.0%}" for h in history]
        print(f"      Seed {seed_idx}: {' → '.join(trajectory)}")
    
    # C) Semantic Dims Recovered
    print("\n[C] SEMANTIC DIMS RECOVERED")
    recoveries = [r["final_dims_recovered"]["mean_pos_recovery"] for r in results]
    print(f"    Mean recovery rate: {np.mean(recoveries):.1%} ± {np.std(recoveries):.1%}")
    
    # Per-class breakdown (from first seed as example)
    print("\n    Per-class breakdown (seed 0):")
    for name, data in results[0]["final_dims_recovered"]["per_class"].items():
        print(f"      {name}: {data['pos_recovered']}/{data['pos_total']} dims, "
              f"discrim_neg={data['discriminative_negatives']}")
    
    # Overall verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    containment_pass = np.mean(containments) > 0.7 and np.std(containments) < 0.15
    purity_pass = np.mean(purities) > 0.9 and np.std(purities) < 0.1
    recovery_pass = np.mean(recoveries) > 0.6
    
    print(f"\n    Containment consistent: {'PASS' if containment_pass else 'FAIL'} "
          f"(mean={np.mean(containments):.1%}, std={np.std(containments):.1%})")
    print(f"    Purity consistent:     {'PASS' if purity_pass else 'FAIL'} "
          f"(mean={np.mean(purities):.1%}, std={np.std(purities):.1%})")
    print(f"    Dims recovered:        {'PASS' if recovery_pass else 'FAIL'} "
          f"(mean={np.mean(recoveries):.1%})")
    
    all_pass = containment_pass and purity_pass and recovery_pass
    print(f"\n    OVERALL: {'✓ CONVERGENCE CONFIRMED' if all_pass else '✗ CONVERGENCE NOT CONFIRMED'}")
    
    if all_pass:
        print("\n    → Signatures recover structure across seeds.")
        print("    → This is reproducible, not coincidence.")
        print("    → Proceed to fuzzy boundary tests.")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_convergence_test(n_seeds=10)
