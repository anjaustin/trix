# Changelog

All notable changes to TriX are documented here.

---

## [0.7.1] - 2025-12-16

### The FP4 Release

**Core achievement:** *Exact computation in 4 bits. Construction, not training.*

This release adds FP4 support to the TriX Compiler - threshold circuit atoms that are exact by construction, packed into 4-bit format.

### Added

#### FP4 Atoms
- 10 threshold circuit atoms verified at 100% accuracy
- Exact by construction (no training convergence risk)
- Minterm generator for custom atoms

#### FP4 Packing
- Custom 4-bit encoding with lookup tables
- Zero quantization error
- `.fp4` weight file format

#### Compiler Integration
- `TriXCompiler(use_fp4=True)` for FP4 mode
- `FP4Emitter`, `FP4Loader`, `FP4CompiledCircuit`
- End-to-end pipeline tested

### Key Insight

> "Don't train atoms to be exact. Construct them to be exact."

FP4 atoms use threshold circuits with hand-crafted weights:
- Weights: {-1, 0, +1}
- Biases: {-2.5, -1.5, -0.5, 0.5, 1.5}

All values fit in 4-bit encoding. Exactness guaranteed.

### Results

| Circuit | Float32 | FP4 | Status |
|---------|---------|-----|--------|
| Full Adder | 100B | 58B | 100% exact |
| 8-bit Adder | 100B | 58B | 100% exact |

### Documentation

- `docs/FP4_INTEGRATION.md` - Complete FP4 guide
- `docs/FP4_ATOMS_RESULTS.md` - Detailed results
- `notes/ROADMAP_FP4.md` - Development roadmap

---

## [0.7.0] - 2025-12-16

### The Compiler Release

**Core achievement:** *Spec → Decompose → Verify → Compose → Emit. The neural network has become a computer.*

This release introduces the TriX Compiler - a complete toolchain for transforming high-level circuit specifications into verified neural circuits that compute **exactly**.

### Added

#### TriX Compiler (`src/trix/compiler/`)

- **AtomLibrary** (`atoms.py`)
  - Pre-verified atomic operations: AND, OR, XOR, NOT, NAND, NOR, XNOR, SUM, CARRY, MUX
  - Exhaustive verification (100% accuracy required)
  - Truth table-based atom definition
  - Atom serialization and caching

- **CircuitSpec** (`spec.py`)
  - Circuit specification language
  - Wire types: INPUT, OUTPUT, INTERNAL
  - Multi-bit wire support
  - Built-in templates: full_adder, adder_8bit, adder_16bit, adder_32bit

- **Decomposer** (`decompose.py`)
  - Circuit decomposition into atoms
  - Dependency graph analysis
  - Topological sort for execution order

- **Verifier** (`verify.py`)
  - Atom verification to 100% accuracy
  - Parallel verification support
  - Exhaustive circuit verification with oracle functions

- **Composer** (`compose.py`)
  - Tile allocation (Hollywood Squares model)
  - Route generation
  - Signature generation for content-addressable routing
  - CircuitExecutor for runtime execution

- **Emitter** (`emit.py`)
  - TrixConfig generation (.trix.json)
  - Weight file emission
  - Manifest with checksums
  - TrixLoader for loading compiled circuits

- **TriXCompiler** (`compiler.py`)
  - Main compiler orchestrating full pipeline
  - Template support
  - compile_and_test helper

#### Demo

- **`scripts/demo_compiler.py`** - Full demonstration of compiler capabilities

#### Documentation

- **`src/trix/compiler/README.md`** - Compiler documentation
- **`src/trix/compiler/CHANGELOG.md`** - Compiler changelog
- **`notes/mesa_reflection_*.md`** - Architectural reflections

### Key Results

| Circuit | Atoms | Tiles | Verification |
|---------|-------|-------|--------------|
| Full Adder | 2 | 2 | 100% (8/8 cases) |
| 8-bit Adder | 2 | 16 | 100% (all arithmetic) |
| 16-bit Adder | 2 | 32 | 100% |
| Custom Circuits | Variable | Variable | 100% required |

### The Pipeline

```
┌─────────┐    ┌───────────┐    ┌────────┐    ┌─────────┐    ┌──────┐
│  Spec   │ -> │ Decompose │ -> │ Verify │ -> │ Compose │ -> │ Emit │
└─────────┘    └───────────┘    └────────┘    └─────────┘    └──────┘
     │              │               │              │             │
 CircuitSpec   Atom Types      100% Exact     Topology      Files
```

### Usage

```python
from trix.compiler import TriXCompiler

compiler = TriXCompiler()
result = compiler.compile("adder_8bit")

# Execute
inputs = {"A[0]": 1, "B[0]": 1, "Cin": 0, ...}
outputs = result.execute(inputs)

# Emit to files
result = compiler.compile("adder_8bit", output_dir="./output")
```

### Theory

The compiler implements the "Neural Von Neumann" architecture discovered through analysis of:

- **TriX** - Tile specialization and routing
- **FLYNNCONCEIVABLE** - Neural networks as exact CPUs (460,928 cases, 100% accuracy)
- **Hollywood Squares OS** - Compositional correctness theorem

**Key insight:** "The routing learns WHEN. The atoms compute WHAT."

### Philosophy

> *"We are not building a Model. We are building a Machine."*

The TriX Compiler proves that neural networks can be **compiled**, not just trained. The weights are the circuit. The inference is the computation. Exactness is inherited from verified components.

---

## [0.6.1] - 2024-12-16

### The Complete FFT Release

**Core achievement:** *A complete spectral subsystem - Forward FFT, Inverse FFT, scales to N=64, 100% round-trip.*

This release completes the FFT register, proving that TriX can execute mathematics with exact precision.

### FFT Register (Complete)

| Component | Status | Result |
|-----------|--------|--------|
| ADDRESS | ✅ | 100% structural learning |
| BUTTERFLY | ✅ | 100% discrete operations |
| STAGE CONTROL | ✅ | 100% routing |
| N=8 REAL FFT | ✅ | 100% composition |
| TWIDDLE FACTORS | ✅ | 100% complex rotation |
| N-SCALING | ✅ | 100% on N=8,16,32,64 |
| FFT/IFFT CLOSURE | ✅ | 100% round-trip |

### Added

#### Twiddle Factors (Complex Rotation)
- **`experiments/fft_atoms/pure_trix_fft_twiddle_v2.py`**: Structural twiddle routing (100%)
- Twiddle selection is structural: `(stage, pos) → W_k`
- Same pattern as ADDRESS - learn structure, execute exactly

#### N-Scaling (8 → 64)
- **`experiments/fft_atoms/pure_trix_fft_nscale_v2.py`**: Scales to any power of 2
- Architecture scales trivially - just add stages
- Results: 100% on N=8, 16, 32, 64

#### FFT/IFFT Closure
- **`experiments/fft_atoms/pure_trix_fft_ifft.py`**: Round-trip verification
- IFFT uses conjugate twiddles + 1/N scaling
- Max error: ~1e-6 (float precision)

### Key Results

```
N=8:  FFT 100%, IFFT 100%, Round-trip error 1.19e-06
N=16: FFT 100%, IFFT 100%, Round-trip error 1.07e-06
N=32: FFT 100%, IFFT 100%, Round-trip error 1.43e-06
N=64: FFT 100%, IFFT 100%, Round-trip error 2.38e-06
```

### Architecture

```
Forward FFT:  W_k = e^{-2πik/N}
Inverse FFT:  W_k = e^{+2πik/N} with 1/N scaling

Fixed Microcode:
  - Twiddle factors (exact complex numbers)
  - Butterfly operations (exact arithmetic)

Learned/Algorithmic Control:
  - Twiddle selection: (stage, pos) → W_k
  - Pairing: i XOR 2^stage
```

### What We Proved

1. **FFT structure IS learnable** (100% on all components)
2. **Once learned, it matches the algorithm exactly**
3. **Pure TriX can execute mathematics**

### Philosophy

> *"This is no longer an experiment. It's infrastructure."*

The FFT subsystem demonstrates that TriX can serve as a neural control plane for mathematical execution - not approximating functions, but executing algorithms.

**CODENAME: ANN WILSON**
- *Barracuda* - The hunt for the solution
- *These Dreams* - Linear-residual attempt
- *Alone* - Discrete ops click
- *What About Love* - Twiddles land
- *Crazy On You* - N-scaling works
- *Never* - Round-trip closure

---

## [0.5.5] - 2024-12-16

### The Pure TriX Release (Mesa 5)

**Core insight:** *Fixed microcode + Learned control = Pure TriX FFT*

This release proves that FFT can be learned with pure TriX - no external organs, no hybrid compute. Fixed operations provide exact arithmetic, routing learns control.

### Added

#### FFT Atoms (Mesa 5: Pure TriX)
- **`experiments/fft_atoms/atom_address.py`**: Structure learning (100%)
- **`experiments/fft_atoms/atom_butterfly.py`**: Arithmetic baseline (0% - expected)
- **`experiments/fft_atoms/pure_trix_fft.py`**: Micro-ops ADD/SUB (100%)
- **`experiments/fft_atoms/pure_trix_butterfly.py`**: Complete butterfly (100%)
- **`experiments/fft_atoms/pure_trix_fft_discrete.py`**: **Full N=8 FFT (100%)**
- **`experiments/fft_atoms/pure_trix_fft_linear.py`**: Linear-residual attempt
- **`experiments/fft_atoms/fft_n8_hybrid.py`**: Hybrid comparison (100%)

#### Documentation
- **`docs/FFT_ATOMS_HYBRID.md`**: Full Mesa 5 documentation with complete journey

### Key Results

#### Full N=8 FFT with Discrete Operations
| Metric | Result |
|--------|--------|
| Operation Selection (SUM path) | 256/256 → Op0 (100%) |
| Operation Selection (DIFF path) | 256/256 → Op1 (100%) |
| Generalization (all ranges) | 100% |
| **Full N=8 FFT** | **100/100 = 100%** |

### The Five Mesas (Complete)

| Mesa | Claim | Status |
|------|-------|--------|
| Mesa 1 | Routing IS computation | ✓ 92% tile purity |
| Mesa 2 | v2 enables partnership | ✓ Surgery, claim tracking |
| Mesa 3 | Paths can be compiled | ✓ 100% A/B agreement |
| Mesa 4 | Temporal binding | ✓ 100% bracket counting |
| **Mesa 5** | **Tiles compute, routing controls** | **✓ 100% pure TriX FFT** |

### The Winning Architecture

```python
# Fixed operations (tiles/microcode)
Op0: (a, b) → a + b  [coeffs: (1, 1)]
Op1: (a, b) → a - b  [coeffs: (1, -1)]

# Learned routing (control)
Router_SUM  → selects Op0 (100%)
Router_DIFF → selects Op1 (100%)
```

**The 6502 parallel is exact:**
- Operations are fixed microcode (like opcodes)
- Routing learns control flow (like instruction sequencing)
- Arithmetic is exact because coefficients are fixed, not learned

### The Journey

1. ADDRESS atom → 100% (TDSR learns structure)
2. BUTTERFLY atom → 0% (TDSR can't do arithmetic)
3. Hybrid → 100% (but needs external organs)
4. "The tiles are programmable, right?" (key question)
5. Pure TriX butterfly → 100% (tiles learn operations)
6. Linear-residual FFT → 0% (coefficient errors compound)
7. **Discrete ops FFT → 100%** (exact arithmetic, learned control)

### Philosophy

> *"Don't learn the arithmetic. Learn WHEN to use each operation."*

The constraint "pure TriX only" forced discovery of the deeper solution.

**CODENAME: ANN WILSON** - *Barracuda, These Dreams, Alone*

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
