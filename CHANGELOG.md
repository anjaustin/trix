# Changelog

All notable changes to TriX are documented here.

---

## [0.4.0] - 2024-12-15

### The SparseLookup Release

**Core insight:** *Routing IS the computation. Wisdom is knowing when not to compute.*

This release introduces `SparseLookupFFN`, a new architecture that emerged from systematic exploration of the hybrid space between HierarchicalTriXFFN and HybridKANFFN. It achieves the best perplexity with the fewest parameters.

### Added

#### New Architecture: SparseLookupFFN
- **`SparseLookupFFN`** - Drop-in FFN replacement where routing selects a direction and splines modulate magnitude. No matrix multiplies in the hot path.
- **`SparseLookupBlock`** - Full transformer block using SparseLookupFFN
- **`TernarySpline2D`** - 2D spline with ternary coefficients ({-1, 0, +1}) and straight-through estimator
- **`FloatSpline2D`** - Float-precision variant for ablation studies

#### Benchmark Infrastructure
- **`scripts/benchmark_ffn.py`** - Head-to-head comparison of HierarchicalTriXFFN, HybridKANFFN, and SparseLookupFFN on TinyShakespeare

#### Tests
- **`tests/test_sparse_lookup.py`** - 22 new tests covering splines, FFN, block, and integration

#### Documentation
- **`notes/00_the_process.md`** - The iteration process that led to SparseLookupFFN
- **`notes/01_raw_thoughts_hybrid.md`** - Initial exploration
- **`notes/02_nodes_of_opportunity.md`** - Candidate architectures evaluated
- **`notes/03_engineering_lens.md`** - Engineering constraints applied
- **`notes/04_convergence.md`** - Final architecture emergence
- **`notes/05_holding_to_the_sun.md`** - Ontological, epistemic, practical, and aesthetic analysis

### Changed

- **README.md** - Updated with SparseLookupFFN as recommended approach, new results table, reproduce instructions
- **Exports** - SparseLookupFFN, SparseLookupBlock, TernarySpline2D now available from `from trix import ...`

### Results

Validated on TinyShakespeare character-level language modeling:

| Model | Params | Val PPL | vs Baseline |
|-------|--------|---------|-------------|
| Sparse-4tiles (v0.3.0) | — | 19.26 | — |
| Hierarchical-16 (v0.3.0) | 826,304 | 17.16 | −10.9% |
| HybridKAN-64 (v0.3.0) | 882,112 | 16.73 | −13.1% |
| **SparseLookup-64 (v0.4.0)** | **366,412** | **16.56** | **−14.0%** |

**SparseLookupFFN: 2.3× fewer parameters, best perplexity.**

### Technical Details

SparseLookupFFN architecture:
```
Input → LayerNorm → [Route to Tile] + [Compress to 2D]
                          ↓                  ↓
                    tile_direction    TernarySpline2D(a,b)
                          ↓                  ↓
                      Output = input + scale × direction
```

Key properties:
- **Routing**: Hierarchical (cluster → tile), signatures derived from direction vectors
- **Compression**: Shared network, d_model → 2 scalars
- **Splines**: 16×16 grid, ternary coefficients, ~200 bytes per tile
- **Directions**: One d_model vector per tile (the "knowledge")

### Migration

To use SparseLookupFFN in existing code:

```python
# Before (v0.3.0)
from trix import HierarchicalTriXFFN
ffn = HierarchicalTriXFFN(d_model=512, num_tiles=16, tiles_per_cluster=4)

# After (v0.4.0)
from trix import SparseLookupFFN
ffn = SparseLookupFFN(d_model=512, num_tiles=64, tiles_per_cluster=8)
```

The API is identical: `output, routing_info, aux_losses = ffn(x)`

---

## [0.3.0] - Prior Release

### Features (as inherited)
- `HierarchicalTriXFFN` - FFN with 2-level hierarchical routing
- `HierarchicalTriXBlock` - Full transformer block
- `SparseTriXFFN` - Simple 4-tile sparse FFN
- `TriXFFN`, `TriXBlock`, `TriXStack` - Classic emergent routing
- `TriXLinear` - Low-level ternary linear layer
- 2-bit kernel with ARM NEON acceleration
- QAT (quantization-aware training) utilities
- 146 tests

### Results (v0.3.0 baseline)
- Hierarchical-16tiles: PPL 17.16 (826K params)
- Sparse-4tiles: PPL 19.26

---

## Philosophy

> *"Don't learn what you can read."* — TriX core principle

> *"Wisdom is knowing when not to compute."* — SparseLookup extension

The progression from v0.3.0 to v0.4.0 represents a deepening of the core insight: if routing can select what to do, maybe routing IS the computation. The spline just modulates how much.

---

[GitHub Repository](https://github.com/anjaustin/trix)
