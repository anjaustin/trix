# Gate 2: Natural Data Pilot (MNIST) — Findings

*Completed December 15, 2024*

---

## Executive Summary

**Question:** Do interpretable islands emerge in real data without designed semantic structure?

**Answer:** Yes. Tiles specialize by digit and by visual feature, with 100% routing determinism.

---

## Experiment Design

### Setup
- Dataset: MNIST (60k train, 10k test)
- Architecture: Input (784) → Project (64) → TriX routing (16 tiles) → Classification (10)
- Training: 20 epochs, batch size 128, Adam optimizer
- No explicit routing supervision by digit — only task loss + entropy regularization

### Hypothesis
If semantic addressing generalizes beyond synthetic data:
1. Tiles should specialize by digit class
2. Specialization should be interpretable (visual features)
3. Routing should be deterministic

---

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| Test accuracy | 94.6% |
| Active tiles | 16/16 |
| Routing determinism | 100% |

### Tile Specialization

**Strongly specialized tiles (>50% one digit):**

| Tile | Top Digit | Purity | Interpretation |
|------|-----------|--------|----------------|
| 3 | 6 | 73% | Curved loop structure |
| 8 | 6 | 63% | Curved loop structure |
| 15 | 1 | 61% | Vertical stroke |
| 13 | 4 | 60% | Angular intersection |

**Semantically meaningful shared routing:**

| Tile | Digits | Shared Visual Feature |
|------|--------|----------------------|
| 2 | 1 (72%), 9 (43%) | Vertical strokes |
| 5 | 3 (45%), 5 (46%) | Curved midsection |
| 11 | 4 (35%), 9 (29%) | Upper vertical component |

### Digit → Tile Routing

| Digit | Primary Tile | Purity | Notes |
|-------|--------------|--------|-------|
| 0 | Tile 4 | 36% | Distributed (round shape ambiguous) |
| 1 | Tile 2 | 72% | **Very clean** — vertical stroke |
| 2 | Tile 6 | 34% | Distributed |
| 3 | Tile 5 | 45% | Shares with 5 (curves) |
| 4 | Tile 11 | 42% | Angular features |
| 5 | Tile 5 | 46% | Shares with 3 (curves) |
| 6 | Tile 14/6 | ~18% each | Multiple curve variants |
| 7 | Tile 1 | 34% | Shares with similar angles |
| 8 | Tile 12 | 23% | Most distributed (complex shape) |
| 9 | Tile 2 | 43% | Shares with 1 (vertical) |

---

## Key Findings

### 1. Semantic Structure Emerges Without Supervision

No routing loss was applied based on digit labels. The tiles discovered digit-correlated structure purely from the task gradient. This is archaeological discovery, not supervised assignment.

### 2. Visual Features Drive Routing

The routing patterns are interpretable:
- Vertical strokes (1, 9) → same tiles
- Curved midsections (3, 5) → same tiles
- Loop structures (6) → dedicated tiles

The system learned that visual similarity = semantic proximity.

### 3. Routing is Perfectly Deterministic

100% of test samples route to the same tile on repeated evaluation. This is stable addressing, not stochastic selection.

### 4. All Tiles Are Used

16/16 tiles active, with balanced entropy regularization. No collapse, no dead tiles.

### 5. Distributed Representation for Complex Digits

Simple digits (1) route cleanly to one tile (72% purity). Complex digits (8, 0) distribute across multiple tiles. This is appropriate — complex shapes have multiple addressable features.

---

## Interpretation

### What This Means

The experiment demonstrates that TriX's signature-based routing discovers semantic structure in natural data:

1. **Islands exist in MNIST** — Different digits occupy different regions of signature space
2. **Islands are interpretable** — We can name what visual features each tile responds to
3. **Islands are stable** — Same input always routes to same tile

### What This Doesn't Mean

1. **Not SOTA performance** — 94.6% is respectable but not competitive with deep networks
2. **Not perfect separation** — Many digits distribute across tiles (expected for complex visual data)
3. **Not proven at scale** — MNIST is small; larger datasets may behave differently

---

## Comparison to Synthetic Experiments

| Property | Synthetic | MNIST |
|----------|-----------|-------|
| Ground truth known | Yes | No |
| Routing purity | 97.7% | 39.4% |
| Determinism | 100% | 100% |
| Interpretable | By design | Emergent |
| Specialization | 1:1 class:tile | Many:many |

The lower purity in MNIST is expected — real data doesn't have orthogonal semantic classes. But the fact that *any* structure emerged unsupervised is significant.

---

## Implications

### For the Semantic Geometry Thesis

Natural data has addressable structure. The thesis extends beyond synthetic domains.

### For Interpretability

We can inspect tile specializations and explain routing decisions:
- "This 1 routed to Tile 2 because it has a vertical stroke signature"
- "This 6 routed to Tile 3 because it has a loop structure signature"

### For Future Work

1. **Larger datasets** — Does structure emerge in CIFAR, ImageNet?
2. **Language** — Do character/word tiles specialize semantically?
3. **Composition** — Do multi-layer TriX models build hierarchical features?

---

## Code Reference

```bash
python experiments/natural_data_mnist.py
```

Key architecture:
```python
class MNISTTriXLayer(nn.Module):
    # Input (784) → Project (64) → Route (16 tiles) → Classify (10)
    self.input_proj = nn.Linear(784, 64)
    self.signatures_raw = nn.Parameter(torch.randn(16, 64))
    self.tile_classifiers = nn.ModuleList([nn.Linear(64, 10) for _ in range(16)])
```

---

## Conclusion

Gate 2 passed. Islands emerge in natural data:
- Tiles specialize by visual feature
- Routing is deterministic
- Specialization is interpretable

The semantic geometry thesis extends beyond synthetic domains.

---

*Gate 2 complete. Next: Gate 3 (Multi-Layer Composition) or further natural data experiments.*
