# Butterfly MatMul: Structured Matrix Multiplication

**One engine, multiple cartridges.**

---

## The Core Insight

FFT and MatMul are the **same structure**:

```
Permutation × Block-Diagonal × Permutation × Block-Diagonal × ...
```

Different blocks give different transforms:

| Blocks | Transform | Complexity |
|--------|-----------|------------|
| `[[1,0],[0,1]]` | Identity | O(N log N) |
| `[[1,1],[1,-1]]` | Hadamard (WHT) | O(N log N) |
| `[[1,W],[1,-W]]` | DFT (FFT) | O(N log N) |
| Learned/designed | Structured MatMul | O(N log N) |

We built the engine for FFT. Now we load different cartridges.

---

## Quick Start

```python
import sys
sys.path.insert(0, 'experiments/matmul')

from butterfly_matmul import identity_butterfly, hadamard_butterfly
import numpy as np

# Identity matrix via butterfly
N = 8
net = identity_butterfly(N)
M = net.as_matrix()
print(f"Identity error: {np.max(np.abs(M - np.eye(N)))}")
# Output: Identity error: 0.0

# Hadamard matrix via butterfly
net = hadamard_butterfly(N)
M = net.as_matrix()
# Matches scipy.linalg.hadamard exactly!
```

---

## Architecture

### Butterfly Layer

One stage of butterfly computation:

```
┌─────────────────────────────────────────────────────────────────┐
│                      BUTTERFLY LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:  x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]       │
│                                                                 │
│  Pairing (XOR with stride):                                    │
│    stride=1: (0,1), (2,3), (4,5), (6,7)                        │
│    stride=2: (0,2), (1,3), (4,6), (5,7)                        │
│    stride=4: (0,4), (1,5), (2,6), (3,7)                        │
│                                                                 │
│  Block operation on each pair:                                  │
│    [y_i]   [b00 b01] [x_i]                                     │
│    [y_j] = [b10 b11] [x_j]                                     │
│                                                                 │
│  Output: y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Butterfly Network

Multiple stages = full transform:

```
Input → Stage 0 → Stage 1 → Stage 2 → Output
        (stride=1) (stride=2) (stride=4)
```

For N inputs: log₂(N) stages, O(N log N) total operations.

---

## Block Opcodes

Like twiddle opcodes for FFT, we have **block opcodes** for MatMul:

### Ternary Block Classification

All 81 ternary 2×2 matrices (entries in {-1, 0, +1}):

| Category | Count | Examples |
|----------|-------|----------|
| Identity-like | 2 | `±[[1,0],[0,1]]` |
| Swap-like | 2 | `±[[0,1],[1,0]]` |
| Hadamard-like | 12 | `[[1,1],[1,-1]]`, `[[1,-1],[1,1]]` |
| Projection | 32 | `[[1,0],[0,0]]`, `[[1,1],[0,0]]` |
| Invertible | 32 | Various |

### Named Opcodes

```python
BLOCK_OPCODES = {
    'I':    [[1, 0], [0, 1]],     # Identity
    'SWAP': [[0, 1], [1, 0]],     # Swap
    'H+':   [[1, 1], [1, -1]],    # Hadamard
    'H-':   [[1, -1], [1, 1]],    # Hadamard variant
    'D+':   [[1, 0], [0, -1]],    # Diagonal flip
    # ... more
}
```

---

## Monarch Matrices

Generalization with larger blocks:

```
M = (B₁ ⊗ I_q) × P × (I_p ⊗ B₂)

For N = p × q:
1. Apply p×p blocks to q groups
2. Permute (transpose)
3. Apply q×q blocks to p groups
```

Complexity: O(N × (p + q)) instead of O(N²)

For N=1024 with p=q=32: **16× faster** than dense.

---

## Verified Results

| Test | Result |
|------|--------|
| Identity N=8 | **0.00 error** |
| Identity N=16 | **0.00 error** |
| Hadamard N=8 | **0.00 error** |
| Hadamard N=16 | **0.00 error** |
| Hadamard = WHT | **Exact match** |
| Monarch permutation | **Correct pattern** |

---

## Connection to FFT

The butterfly MatMul uses **identical structure** to our FFT:

```python
# FFT pairing
partner = pos ^ (2 ** stage)

# Butterfly MatMul pairing
partner = pos ^ stride  # where stride = 2 ** stage

# SAME!
```

Test confirms: `test_butterfly_uses_same_pairing_as_fft` passes.

---

## TriX Butterfly MLP (Prototype)

Replace dense FFN with butterfly structure:

```python
# Dense FFN: O(d²)
# Linear(d, 4d) → GELU → Linear(4d, d)

# Butterfly FFN: O(d log d)
# ButterflyUp → GELU → ButterflyDown
```

Current prototype reduces parameters but needs refinement for proper expansion/contraction.

---

## Why This Matters

### The Vision

If we can replace:
- Attention → FFT/spectral (done!)
- MLP → Butterfly MatMul (in progress)

We get a **MatMul-free Transformer**:
- O(N log N) sequence mixing
- O(N log N) channel mixing
- All routing + local ops

### Structured vs Random

| Matrix Type | Butterfly Error |
|-------------|-----------------|
| Identity | 0.00 |
| Hadamard | 0.00 |
| DFT | 0.00 (with twiddles) |
| Random | ~1.35 relative error |

**Key insight:** Random matrices don't decompose cleanly. But structured matrices (the ones that matter for efficient inference) do.

---

## Files

| File | Purpose |
|------|---------|
| `experiments/matmul/butterfly_matmul.py` | Implementation |
| `tests/test_butterfly_matmul.py` | 16 rigorous tests |
| `docs/BUTTERFLY_MATMUL.md` | This document |
| `notes/matmul_*.md` | Exploration journals |

---

## The Punchline

```
FFT:    Route → Twiddle → Route → Twiddle → ...
MatMul: Route → Block   → Route → Block   → ...
Both:   Route → Local   → Route → Local   → ...
```

**We built the engine. Now we load any cartridge.**

---

*MatMul is FFT with different blocks.*
