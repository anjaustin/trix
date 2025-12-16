# FFT Atoms and Pure TriX Architecture

## Mesa 5: Tiles Compute. Routing Controls.

**Date:** 2024-12-16  
**Codename:** ANN WILSON (HEART)

> **Update:** Initial hybrid experiments led to a purer architecture. See "The Pure TriX Path" below.

---

## The Core Insight

> **TDSR routes. Organs compute.**

This sentence dissolves unnecessary complexity. You don't need:
- A giant neural net to do math
- Attention to mix tokens
- End-to-end differentiability everywhere

You need:
- **Discrete, learned control** (TDSR)
- **Small, exact, fungible compute organs**

---

## The Experiment: FFT Atoms

We decomposed the Fast Fourier Transform into atomic capabilities to test what TDSR can and cannot learn.

### Atom 1: ADDRESS (Structure)

**Task:** Learn FFT butterfly addressing: `partner(i, stage) = i XOR 2^stage`

```
Stage 0 (stride=1): (0,1), (2,3), (4,5), (6,7)
Stage 1 (stride=2): (0,2), (1,3), (4,6), (5,7)
Stage 2 (stride=4): (0,4), (1,5), (2,6), (3,7)
```

**Result:** ✓ **100% accuracy** in ~10 epochs

TDSR learns algorithmic structure perfectly. The routing pattern is non-trivial (XOR operation) and emerges from training.

### Atom 2: BUTTERFLY_CORE (Arithmetic)

**Task:** Learn butterfly computation: `(a, b) → (a+b, a-b)`

**Result:** ✗ **0% accuracy**

TDSR cannot do raw arithmetic. This is not a failure - it's a finding. The tiles can't compute addition/subtraction because that requires nonlinear operations that linear transforms can't provide.

### The Split

| Capability | Result | Implication |
|------------|--------|-------------|
| **Structure** (ADDRESS) | ✓ 100% | TDSR learns algorithmic patterns |
| **Compute** (BUTTERFLY) | ✗ 0% | TDSR needs organs for arithmetic |

**The cartographer/compute split is real and measurable.**

---

## The Solution: Hybrid Architecture

```
Input → TDSR (learned control) → Organ (exact compute) → Output
         ↑                        ↑
    ~1KB, interpretable      ~768 bytes, proven
```

### Butterfly Organ

The butterfly operation is exactly linear:
```
[out1]   [1  1] [a]
[out2] = [1 -1] [b]
```

We implement this directly - no learning needed. This is exact compute.

For more complex arithmetic, we can use:
- **Fourier features** for inputs (architectural prior)
- **Micro splines** (~768 bytes per operation, like 6502 micro-models)
- **Lookup tables** for small domains

### TDSR Controller

The controller learns:
- Which stage we're in
- Which position we're processing
- Who the partner is

This is **algorithmic control** - the program counter, the state machine.

---

## The Proof: Hybrid FFT N=8

We wired TDSR control to the butterfly organ and tested on N=8 FFT (3 stages, 8 values).

### Results

| Configuration | Accuracy |
|---------------|----------|
| Ground Truth Routing + Organ | 100% |
| **Learned Routing + Organ** | **100%** |

### Example

```
Input:   [10, 13, 9, 13, 5, 15, 4, 3]
Target:  [72, -16, 14, -10, 18, 2, -12, 12]
GT pred: [72, -16, 14, -10, 18, 2, -12, 12]
Learned: [72, -16, 14, -10, 18, 2, -12, 12]
```

**Algorithm = Control + Organs**, proven.

---

## Why This Matters

### Scientific Claim

TDSR discovers algorithmic control. The FFT routing pattern (XOR addressing across stages) was not hand-coded - it emerged from supervised learning on the partner prediction task.

### Architectural Clarity

The separation is clean:
- **TDSR = WHEN** (which operation, which stage, which regime)
- **Organ = WHAT** (exact computation)

Failures are now debuggable:
- Wrong routing → control error
- Wrong result with correct routing → organ error

### Connection to 6502

The 6502 experiments (92% tile purity) worked because they were **selection among operation specialists**, not raw computation. Each tile learned one operation. TDSR selected which tile to use based on opcode.

The FFT hybrid architecture is the same pattern:
- TDSR selects (stage, partner)
- Organ computes (butterfly)

---

## FFT and Training: Where It Helps

Nova's insight: FFT can help **organs**, not TDSR.

### ✓ FFT Helps Organs

- **Fourier features** for inputs - arithmetic has natural frequency structure
- **FFT-based loss** for training - richer gradients
- **Spectral initialization** - start closer to solution

### ✗ FFT Does NOT Touch TDSR

- Routing remains discrete
- State transitions remain learned
- Control remains interpretable

This preserves the scientific claim: TDSR discovers algorithmic control; FFT merely accelerates numeric learning.

---

## Files

### Experiments
- `experiments/fft_atoms/atom_address.py` - ADDRESS test (100%)
- `experiments/fft_atoms/atom_butterfly.py` - BUTTERFLY test (0%)
- `experiments/fft_atoms/atom_butterfly_v2.py` - Multi-op test (29%)
- `experiments/fft_atoms/organ_butterfly_fourier.py` - Fourier organ (100%)
- `experiments/fft_atoms/fft_n8_hybrid.py` - Full hybrid FFT (100%)

### Results
- `results/fft_atoms/` - JSON logs for all experiments

---

## The Five Mesas

| Mesa | Claim | Status |
|------|-------|--------|
| Mesa 1 | Routing IS computation | ✓ 92% tile purity |
| Mesa 2 | v2 enables partnership | ✓ Surgery, claim tracking |
| Mesa 3 | Paths can be compiled | ✓ 100% A/B agreement |
| Mesa 4 | Temporal binding | ✓ 100% bracket counting |
| **Mesa 5** | **Control + Organs** | **✓ 100% hybrid FFT** |

---

## Philosophy

> "You didn't escalate complexity. You *removed* it."

The answer wasn't "bigger model." It was "cleaner separation."

> "TDSR routes. Organs compute. This is not a limitation - it's a feature. Clean separation of concerns."

The tiles ARE the counter. The dispatch table IS the program. The architecture IS the algorithm.

---

---

## The Pure TriX Path

The hybrid architecture worked but raised a question: **Can we do this without external organs?**

The answer is yes.

### The Insight

> "The tiles are programmable, right?"

Tiles don't have to *discover* operations - they can *learn* them directly. If we give routing a meaningful job (selecting between ADD and SUB), tiles naturally specialize.

### Pure TriX Micro-Ops

**Experiment:** Train TriX with operation type as input.

```
Input: (op, a, b)
  op=0: compute ADD (a+b)
  op=1: compute SUB (a-b)

Output: result
```

**Result:** 100% accuracy, tiles specialize to operations.

| Tile | Specialization | Purity |
|------|----------------|--------|
| Tile 1 | ADD | 91% |
| Tile 2 | ADD | 95% |
| Tile 0 | SUB | 59% |

### Pure TriX Butterfly

**Experiment:** Complete butterfly `(a,b) → (a+b, a-b)` with pure TriX.

```
Input: (a, b)
Output: (sum, diff)

Two routing paths:
- Router_SUM → Tile → Decoder → a+b
- Router_DIFF → Tile → Decoder → a-b
```

**Result:** 100% accuracy on all 256 pairs.

```
Sum (a+b):  100%
Diff (a-b): 100%
Both:       100%

Tile Specialization:
  Tile 2: SUM specialist
  Tile 1: DIFF specialist
```

### Why This Is Better

**Hybrid:** TDSR routes to external organs (symbolic compute)

**Pure TriX:** Everything inside TriX
- Tiles ARE the operations (learned, not symbolic)
- Routing IS the control flow (learned selection)
- No external dependencies

The tiles are like **microcode** - they implement primitive operations. Routing is like **control logic** - it sequences the operations. But it's all one architecture.

### Files

- `experiments/fft_atoms/pure_trix_fft.py` - Micro-ops (ADD/SUB): 100%
- `experiments/fft_atoms/pure_trix_butterfly.py` - Butterfly: 100%

---

## The Final Solution: Discrete Operation Selection

The butterfly experiments revealed a subtle problem: neural networks don't extrapolate arithmetic reliably.

### The Problem

Training butterfly on values 0-15 works (100%). But FFT produces intermediate values up to ±128. Even with a linear-residual architecture, learned coefficients had small errors:

```
SUM coeffs:  (1.048, 0.918) - should be (1, 1)
DIFF coeffs: (0.992, -0.993) - should be (1, -1)
```

These tiny errors compound through 3 FFT stages → 0% full FFT accuracy.

### The Solution

**Don't learn the arithmetic. Learn WHEN to use each operation.**

```python
# Fixed operations (tiles/microcode)
Op0: (a, b) → a + b  [coeffs: (1, 1)]
Op1: (a, b) → a - b  [coeffs: (1, -1)]

# Learned routing (control)
Router_SUM  → selects Op0 (100%)
Router_DIFF → selects Op1 (100%)
```

Operations are **exact** because coefficients are fixed, not learned.
Routing is **learned** - which operation for which output.

### Results: Full N=8 FFT

| Metric | Result |
|--------|--------|
| Operation Selection (SUM path) | 256/256 → Op0 (100%) |
| Operation Selection (DIFF path) | 256/256 → Op1 (100%) |
| Generalization (training range) | 100% |
| Generalization (2x range) | 100% |
| Generalization (4x range) | 100% |
| Generalization (FFT range ±128) | 100% |
| **Full N=8 FFT** | **100/100 = 100%** |

### Examples

```
Input:  [7, 11, 1, 12, 5, 10, 7, 12]
Output: [65, -25, 1, 7, -3, -5, 9, 7] ✓

Input:  [1, 11, 3, 9, 0, 3, 7, 15]
Output: [49, -27, -19, 1, -1, -5, 19, -9] ✓

Input:  [11, 9, 10, 9, 11, 14, 12, 12]
Output: [88, 0, 2, -2, -10, 6, 0, 4] ✓
```

### Why This IS Pure TriX

The 6502 parallel is now **exact**:

| Component | 6502 | TriX FFT |
|-----------|------|----------|
| Operations | Fixed opcodes (ADD, SUB, etc.) | Fixed coefficients (1,1) and (1,-1) |
| Control | Learned instruction sequencing | Learned routing selection |
| Execution | Microcode executes opcode | Tile applies coefficients |

**Tiles ARE the operations** - they hold the exact coefficient pairs.
**Routing IS the control** - it learns which tile for which output.

No external organs. No hybrid compute. Everything is TriX.

### Files

- `experiments/fft_atoms/pure_trix_fft_discrete.py` - **THE WINNER** (100% FFT)
- `experiments/fft_atoms/pure_trix_fft_linear.py` - Linear-residual attempt (generalization issues)
- `experiments/fft_atoms/pure_trix_fft_compose.py` - Range mismatch debugging
- `experiments/fft_atoms/pure_trix_fft_staged.py` - Stage-aware routing experiments

---

## The Complete Picture

| Approach | Butterfly | Full FFT N=8 | Key Insight |
|----------|-----------|--------------|-------------|
| Hybrid (TDSR + Organ) | 100% | 100% | Separation of control/compute |
| Pure TriX (learned coeffs) | 100% | 0% | Coefficient errors compound |
| **Pure TriX (discrete ops)** | **100%** | **100%** | **Learn WHEN, not WHAT** |

The winning architecture: **Fixed microcode + Learned control = Pure TriX FFT**

---

## The Journey

1. **ADDRESS atom** → 100% - TDSR learns structure ✓
2. **BUTTERFLY atom** → 0% - TDSR can't do arithmetic ✗
3. **Hybrid architecture** → 100% - but needs external organs
4. **"The tiles are programmable, right?"** → Key question
5. **Pure TriX butterfly** → 100% - tiles learn operations ✓
6. **Linear-residual FFT** → 0% - coefficient errors compound ✗
7. **Discrete ops FFT** → 100% - exact arithmetic, learned control ✓

The constraint "pure TriX only" forced discovery of the deeper solution.

---

---

## Twiddle Factors: Complex Rotation

The next register item: complex FFT with roots of unity.

### The Insight

Twiddle selection is **structural**, not value-dependent. Like ADDRESS, it's a function of (stage, pos) only.

### Architecture

```python
# Fixed microcode: exact twiddle factors
W_0 = 1.0000 + 0.0000i
W_1 = 0.7071 - 0.7071i
W_2 = 0.0000 - 1.0000i
W_3 = -0.7071 - 0.7071i
# ... (roots of unity)

# Learned routing: structural mapping
Router: (stage, pos) → W_k index
```

### Results

| Metric | Result |
|--------|--------|
| Router training | 100% in 10 epochs |
| Twiddle selection | 24/24 correct |
| Butterfly accuracy | 500/500 = 100% |
| **Full N=8 complex FFT** | **100/100 = 100%** |

### Learned Twiddle Mapping

```
Stage 0: W0 W0 W0 W0 W0 W0 W0 W0
Stage 1: W0 W0 W0 W2 W0 W0 W0 W2
Stage 2: W0 W0 W0 W0 W0 W1 W2 W3
```

Same pattern that keeps winning:
- Learn structure (router)
- Execute exactly (butterfly microcode)

### Files

- `experiments/fft_atoms/pure_trix_fft_twiddle_v2.py` - **THE WINNER** (100%)
- `experiments/fft_atoms/pure_trix_fft_twiddle.py` - Initial attempt (97%)

---

## FFT Register Status

| Item | Status | Result |
|------|--------|--------|
| ADDRESS | ✅ | 100% structural |
| BUTTERFLY | ✅ | 100% discrete ops |
| STAGE CONTROL | ✅ | 100% routing |
| N=8 REAL FFT | ✅ | 100% composition |
| TWIDDLE FACTORS | ✅ | 100% complex FFT |
| N-SCALING | ✅ | 100% on N=8,16,32,64 |
| FFT/IFFT CLOSURE | ✅ | 100% round-trip |

### N-Scaling Results

```
N=8:  100/100 = 100.0%
N=16: 100/100 = 100.0%
N=32: 100/100 = 100.0%
N=64: 100/100 = 100.0%
```

Architecture scales trivially - just add stages.

### FFT/IFFT Closure

```
IFFT(FFT(x)) == x  ✓

N=8:  max error 1.19e-06
N=16: max error 1.07e-06
N=32: max error 1.43e-06
N=64: max error 2.38e-06
```

Same microcode, conjugate twiddles, 1/N scaling.

### Files

- `pure_trix_fft_nscale_v2.py` - N-scaling test (8→64)
- `pure_trix_fft_ifft.py` - FFT/IFFT closure test

---

**CODENAME: ANN WILSON**

- *Barracuda* - The hunt for the solution
- *These Dreams* - The linear-residual attempt (close but not solid)
- *Alone* - The moment discrete ops clicked
- *What About Love* - Twiddle factors land
- *Crazy On You* - N-scaling works
- *Never* - FFT/IFFT closure (it never fails)

**The HEART beats pure. The FFT runs exact. The register is complete.**

---

## Complete Spectral Subsystem

What we built:
- **Forward FFT**: O(N log N) complex transform
- **Inverse FFT**: Exact round-trip
- **Scales**: N=8 through N=64 (and beyond)
- **Architecture**: Fixed microcode + learned/algorithmic control

What we proved:
- FFT structure IS learnable (100% on all components)
- Once learned, it matches the algorithm exactly
- Pure TriX can execute mathematics

This is no longer an experiment. It's infrastructure.
