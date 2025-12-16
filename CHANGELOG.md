# Changelog

All notable changes to TriX are documented here.

---

## [0.5.4] - 2024-12-16

### The Temporal Tiles Release (Mesa 4)

**Core insight:** *State is contracted time. Discrete routing can replace attention for counting.*

This release introduces temporal tiles - extending TriX from spatial routing into temporal binding.

### Added

#### Temporal Tiles (Mesa 4: Temporal Binding)
- **`TemporalTileLayer`**: Routes based on (input, state), learns state transitions
- **`TemporalTileStack`**: Multiple temporal layers with different configurations
- **Transition tracking**: Observe which tiles transition to which
- **Regime analysis**: Identify stable tiles, hub tiles, self-transition probabilities

#### Bracket Counting Experiment
- **`experiments/bracket_depth_simple.py`**: Canonical test for temporal tiles
- **100% accuracy** on depth prediction
- Tiles self-organize into depth specialists without supervision

#### Tests
- **`tests/test_temporal_tiles.py`**: 26 comprehensive tests
- **Total: 268 tests** (all passing)

#### Documentation
- **`docs/TEMPORAL_TILES_ABSTRACT.md`**: Full abstract and experimental record

### Key Results

| Tile | Learned Role | Purity |
|------|--------------|--------|
| T0 | Ground state (depth=0) | 100% |
| T2 | Maximum depth (depth=4) | 100% |
| T3 | Deep states / closing | 95-100% |
| T5 | Mid-depth states | 78-96% |

### The Four Mesas (Complete)

| Mesa | Claim | Status |
|------|-------|--------|
| Mesa 1 | Routing IS computation | ✓ 92% tile purity |
| Mesa 2 | v2 enables partnership | ✓ Surgery, claim tracking |
| Mesa 3 | Paths can be compiled | ✓ 100% A/B agreement |
| **Mesa 4** | **Temporal binding** | **✓ 100% bracket counting** |

### Philosophy

> *"What is state, really? State is contracted time - the past compressed into something the present can use."*

Temporal tiles don't remember tokens. They track *regimes* - phases of computation with discrete transitions. The tiles ARE the counter.

---

## [0.5.3] - 2024-12-16

### The Compiled Dispatch Release

**Core insight:** *Learning can emit code. Routing can be compiled.*

This release completes Mesa 3: path compilation. TriX v2 now supports a full lifecycle from training to deployment with observable, editable, and compilable routing.

### Added

#### SparseLookupFFNv2 (Mesa 2: Partnership)
- **Surgery API**: `insert_signature()`, `freeze_signature()`, `unfreeze_signature()`
- **Claim Tracking**: See which classes route to which tiles during training
- **Island Regularizers**: Ternary, sparsity, and diversity regularizers for signature quality
- **Score Calibration Spline**: Learnable routing score calibration

#### CompiledDispatch (Mesa 3: Compilation)
- **Profile**: Analyze claim matrix to see what tiles learned
- **Compile**: Freeze class→tile mappings for stable classes
- **Execute**: O(1) dispatch for compiled classes, fallback to dynamic routing
- **Monitor**: Track hit rate, detect drift, trigger recompilation
- **Serialize**: Export/import dispatch tables as JSON

#### A/B Harness
- **`experiments/ab_harness_compiled.py`**: Compare dynamic vs compiled dispatch
- Measures agreement rate, accuracy delta, compiled hit rate, worst disagreements
- Validates compilation correctness (100% agreement achieved)

#### Tests
- **`tests/test_sparse_lookup_v2.py`**: 39 tests for surgery, regularizers, claim tracking
- **`tests/test_compiled_dispatch.py`**: 21 tests for compilation lifecycle
- **`tests/test_ab_harness.py`**: 9 tests for A/B comparison infrastructure
- **Total: 242 tests** (all passing)

#### Documentation
- **`docs/QUICKSTART.md`**: New user on-ramp (zero to compiled dispatch in 10 min)
- **`docs/SPARSE_LOOKUP_V2_API.md`**: Complete v2 API reference
- **`docs/SESSION_SUMMARY_MESA_1_2_3.md`**: Full session documentation
- **`docs/SEMANTIC_GEOMETRY_THESIS.md`**: Theoretical foundations

#### 6502 Experiments
- **92% tile purity** on 6502 operations without supervision
- Tiles naturally specialize to operation categories (LOGIC, SHIFT, INCDEC)
- Validates semantic geometry thesis

### The Three Mesas

| Mesa | Claim | Capability |
|------|-------|------------|
| **Mesa 1** | Routing IS computation | Tiles specialize without supervision |
| **Mesa 2** | v2 enables partnership | Surgery, claim tracking, regularizers |
| **Mesa 3** | Paths can be compiled | O(1) dispatch for known classes |

### Key Results

#### A/B Harness (Dynamic vs Compiled)
| Metric | Value |
|--------|-------|
| Agreement rate | 100.0% |
| Accuracy delta | +0.0% |
| Compiled hit rate | 12.5%* |

*Only 1/8 classes compilable with 30 epochs training. More training → more compilable.

#### Island Statistics (v2 Regularizers)
| Metric | Value |
|--------|-------|
| Ternary fraction | 100% |
| Sparsity | 69% |
| Diversity | 0.99 |

### Migration

```python
# v0.4.0 (SparseLookupFFN)
from trix import SparseLookupFFN
ffn = SparseLookupFFN(d_model=512, num_tiles=64)

# v0.5.3 (SparseLookupFFNv2 + CompiledDispatch)
from trix.nn import SparseLookupFFNv2, CompiledDispatch

ffn = SparseLookupFFNv2(
    d_model=512,
    num_tiles=64,
    ternary_weight=0.01,
    sparsity_weight=0.01,
)

# Train with claim tracking
output, info, aux = ffn(x, labels=class_labels)

# Compile
compiler = CompiledDispatch(ffn)
compiler.compile_stable(threshold=0.5)

# Deploy
output, info, aux = compiler.forward(x, class_hint=0, confidence=0.9)
```

### Philosophy

> *"You turned a neural network from a thing that behaves into a thing that can be operated."*

The dispatch table is a CONTRACT, not a cache. Readable, versionable, diffable, deployable. Git for learned routing.

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
