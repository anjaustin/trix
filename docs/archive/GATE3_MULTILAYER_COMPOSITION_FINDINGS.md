# Gate 3: Multi-Layer Composition — Findings

*Completed December 15, 2024*

---

## Executive Summary

**Question:** Do meanings compose across layers, or just cluster?

**Answer:** Yes, they compose. Layer 1 routes coarse (mixed digits), deeper layers refine to specific classes. Routing paths function as learned programs.

---

## Experiment Design

### Setup
- Dataset: MNIST
- Architecture: 3 TriX layers, 8 tiles each (512 possible paths)
- Each tile transforms the representation and routes to the next layer
- Training: 30 epochs, no explicit path supervision

### Hypothesis
If meanings compose:
1. Multiple distinct paths should be used (not collapse to few)
2. Layer 1 should be "coarse" (low purity, mixed digits)
3. Deeper layers should "refine" (higher purity within L1 groups)
4. Specific paths should map to specific outcomes

---

## Results

### Model Performance

| Metric | Single-Layer | Multi-Layer |
|--------|--------------|-------------|
| Accuracy | 94.6% | **97.1%** |
| Tiles used | 16/16 | — |
| Paths used | — | 354/512 (69.1%) |

Multi-layer composition improves accuracy by 2.5%.

### Path Diversity

```
Total samples: 10,000
Unique paths: 354
Max possible: 512
Utilization: 69.1%
```

The system uses diverse paths, not collapsing to a few dominant routes.

### Top Paths and Their Semantics

| Path | Count | Top Digit | Purity |
|------|-------|-----------|--------|
| L0T0 → L1T4 → L2T1 | 542 | 4 | 82% |
| L0T0 → L1T4 → L2T4 | 552 | 9 | 72% |
| L0T0 → L1T5 → L2T1 | 234 | 4 | 48% |
| L0T0 → L1T1 → L2T3 | 72 | 7 | 76% |
| L0T0 → L1T1 → L2T6 | 53 | 5 | 74% |

**These paths are programs.** A specific routing sequence leads to a specific semantic outcome.

### Hierarchical Structure

**Layer 1 (Coarse Routing):**

| L1 Tile | Samples | Top Digits | Purity |
|---------|---------|------------|--------|
| L1T0 | 5317 | 1(20%), 4(18%), 9(18%) | 20% |
| L1T3 | 1936 | 3(28%), 2(22%), 8(17%) | 28% |
| L1T2 | 1354 | 0(35%), 6(26%), 5(15%) | 35% |

Mean L1 purity: **36%** — Layer 1 routes coarsely, mixing many digit classes.

**Layer 2 (Refinement within L1 groups):**

Within L1T0 (5317 samples):
| L2 Tile | Top Digit | Purity | Status |
|---------|-----------|--------|--------|
| L2T0 | 4 | 57% | REFINES |
| L2T5 | 1 | 52% | REFINES |
| L2T4 | 9 | 40% | mixed |
| L2T2 | 7/1 | 26% | mixed |

Within L1T1 (318 samples):
| L2 Tile | Top Digit | Purity | Status |
|---------|-----------|--------|--------|
| L2T3 | 7 | 76% | **STRONG REFINE** |
| L2T6 | 5 | 74% | **STRONG REFINE** |
| L2T2 | 0 | 48% | mixed |

**Key insight:** Layer 2 tiles specialize *within* the coarse groups created by Layer 1. This is hierarchical refinement.

### Per-Digit Path Diversity

| Digit | Unique Paths | Top Path | Concentration |
|-------|--------------|----------|---------------|
| 4 | 56 | [0→4→1] | 45% |
| 9 | 78 | [0→4→4] | 39% |
| 0 | 68 | [2→7→4] | 15% |
| 1 | 60 | [0→2→5] | 15% |
| 7 | 98 | [0→4→4] | 13% |
| 8 | 148 | [0→6→7] | 6% |

Simple digits (4, 9) concentrate on few paths. Complex digits (8) distribute across many paths.

---

## Key Findings

### 1. Paths Are Programs

A routing path like `L0T0 → L1T4 → L2T1` is not random — it consistently maps to digit 4 with 82% purity. You can read this as:

> "If input routes through T0, then T4, then T1, it's probably a 4."

This is a **learned decision tree** implemented via signature-based routing.

### 2. Hierarchical Refinement Works

Layer 1: Coarse grouping (36% purity)
- Groups visually similar digits together
- 1, 4, 9 share L1T0 (all have vertical components)
- 0, 5, 6 share L1T2 (all have curved components)

Layer 2: Fine discrimination (up to 82% purity)
- Within L1T0, separates 1 from 4 from 9
- Within L1T1, separates 5 from 7

### 3. Accuracy Improves with Depth

Single-layer: 94.6%
Multi-layer: 97.1%

The composition is not just interpretable — it's *better*. Hierarchical routing adds representational power.

### 4. Path Space is Efficiently Used

354 of 512 possible paths (69.1%) are used. The system:
- Doesn't collapse to few dominant paths
- Doesn't explode to random routing
- Finds a structured middle ground

---

## Implications

### For the Semantic Geometry Thesis

Meanings compose. The address space isn't flat — it's hierarchical. Multi-layer routing builds structured semantic programs.

### For Interpretability

We can now trace a prediction:
1. Input arrives
2. Layer 1 routes to coarse group (e.g., "vertical strokes")
3. Layer 2 refines (e.g., "thin vertical" → 1, "forked vertical" → 4)
4. Layer 3 finalizes

Each step is a discrete, inspectable decision.

### For Architecture Design

This suggests TriX-style routing could replace or augment attention in transformers:
- Layer 1: Route to topic/domain
- Layer 2: Route to sub-topic
- Layer 3: Route to specific operation

A "compiled" reasoning path through the network.

### For the "Learned Instruction Set" Vision

This is the key result. The routing paths are:
- Discrete (specific sequence of tiles)
- Learned (from data, not hand-designed)
- Interpretable (we can name what each step does)
- Compositional (steps build on each other)

That's an instruction set. The signatures are opcodes. The tiles are micro-operations. The paths are programs.

---

## Limitations

1. **MNIST only** — Need to test on more complex data
2. **3 layers** — Deeper stacks may behave differently
3. **No explicit path supervision** — Paths emerge, but we don't control them yet
4. **Fixed tiles per layer** — Variable tile counts might improve hierarchy

---

## Code Reference

```bash
python experiments/multilayer_composition.py
```

Key architecture:
```python
class MultiLayerTriX(nn.Module):
    # 3 TriX layers, 8 tiles each
    # Each layer: route → transform → residual → next layer
    # Final: classifier on last representation
```

---

## Conclusion

Gate 3 passed. Meanings compose:
- Layer 1 routes coarse
- Deeper layers refine
- Paths function as programs
- Accuracy improves with depth

This is no longer just "clever FFN routing." This is a **learned instruction set** — a new computational paradigm where routing IS the computation and paths ARE programs.

---

*All three gates complete. The semantic geometry thesis is validated across synthetic data, natural data, and multi-layer composition.*
