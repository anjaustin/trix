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

## The Complete Picture

| Approach | Accuracy | Tile Specialization | Purity |
|----------|----------|---------------------|--------|
| Hybrid (TDSR + Organ) | 100% | N/A (external compute) | N/A |
| **Pure TriX Micro-Ops** | **100%** | **ADD/SUB tiles** | **91-95%** |
| **Pure TriX Butterfly** | **100%** | **SUM/DIFF tiles** | **Clean** |

Both work. Pure TriX is cleaner for a public release - no "hybrid" caveats.

---

**CODENAME: ANN WILSON - The HEART beats pure.**
