# TriX Research Summary: Compiled Neural Transforms

**From "train and hope" to "compose and prove"**

---

## Abstract

We present a compilation approach to neural transforms that achieves:

1. **Walsh-Hadamard Transform**: 100% exact via FP4 threshold circuit routing
2. **Discrete Fourier Transform**: 0.00 error via twiddle opcodes (no runtime trig)
3. **General pattern**: Structural routing + Fixed microcode = Verified computation

Key insight: Neural computation can be **compiled**, not just trained.

---

## The Core Discovery

### What We Built vs What We Thought We Built

Initial goal: Compile FFT to FP4 neural circuits.

Discovery: Our XOR-based pairing structure implements **Walsh-Hadamard Transform**, not DFT.

```
partner = pos XOR 2^stage  →  WHT (Hadamard family)
                           ≠  DFT (Fourier family)
```

This was not a bug. It was a revelation about what the structure computes.

### The Two Transform Families

| Transform | Structure | Twiddles | Domain |
|-----------|-----------|----------|--------|
| WHT | XOR pairing | None (±1 only) | Real |
| DFT | Cooley-Tukey groups | Complex roots of unity | Complex |

Both compile to the same pattern: **routing selects, microcode computes**.

---

## Technical Contributions

### 1. FP4 Threshold Circuit Compilation

Routing functions become boolean circuits:

```
IS_UPPER(stage, pos) → 0 or 1
PARTNER(stage, pos) → partner_index  
TWIDDLE(stage, pos) → twiddle_index
```

Each compiled to FP4 threshold circuits:
- Weights: {-1, 0, +1}
- Biases: {-2.5, -1.5, -0.5, 0.5, 1.5}
- Verification: 100% on all truth table entries

### 2. Twiddle Opcodes

Eliminate runtime trigonometry:

```python
# Before (runtime computation)
wm = np.cos(-2*pi/m) - i*np.sin(-2*pi/m)

# After (fixed microcode)
wt = TWIDDLE_OPS[tw_idx](t_re, t_im)
```

For N=8: 8 opcodes with algebraic constants (1, -1, i, -i, ±√½).

### 3. Structural Routing Formula

Twiddle selection is not learned:

```python
tw_idx = (j * (N // m)) % N
```

This is pure structure. No training. No approximation.

---

## Results

### Walsh-Hadamard Transform

| N | IS_UPPER | PARTNER | Accuracy |
|---|----------|---------|----------|
| 8 | 100% (2 layers) | 100% (2 layers/bit) | **100% exact** |
| 16 | 100% (2 layers) | 100% (2 layers/bit) | **100% exact** |
| 32 | 100% (2 layers) | 100% (2 layers/bit) | **100% exact** |

Self-inverse property verified: WHT(WHT(x)) = N·x

### Discrete Fourier Transform

| N | Twiddle Circuits | Opcodes | Error vs NumPy |
|---|------------------|---------|----------------|
| 8 | 100% verified | 4 used | **0.00** |
| 16 | 100% verified | 8 used | ~2e-15 |

VGem's guards pass:
- No `np.cos`, `np.sin`, `exp` in execution path
- All required opcodes exercised

---

## The Compilation Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPILED TRANSFORM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STRUCTURAL ROUTING (FP4 Threshold Circuits)                   │
│  ├── IS_UPPER(stage, pos) → assignment decision                │
│  ├── PARTNER(stage, pos) → pairing index                       │
│  └── TWIDDLE(stage, pos) → opcode selection                    │
│                                                                 │
│  FIXED MICROCODE (No Learning)                                 │
│  ├── Butterfly: (a+b, a-b)                                     │
│  └── Twiddle: (re·c - im·s, re·s + im·c)                      │
│                                                                 │
│  RESULT: Deterministic, Verifiable, Exact                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implications

### For Neural Computation

Traditional neural networks:
- Train weights to approximate functions
- Hope generalization works
- Accept some error

Compiled neural circuits:
- Construct weights for exact functions
- Verify on all inputs
- Guarantee correctness

### For Transform Libraries

Standard FFT:
- Runtime trig computation
- Accumulated floating-point error
- Complex, optimized code

Compiled FFT:
- Fixed opcode table
- Direct lookup + multiply
- Simple, verifiable code

### For Hardware

Twiddle opcodes map directly to silicon:
- ROM for constants
- Fixed MAC units
- No trig hardware

---

## The Broader Pattern

This work validates the TriX thesis:

```
Atomic Decomposition + Tile Specialization + Verified Composition = Neural CPU
```

Applied to transforms:
- **Atoms**: Butterfly, twiddle multiply
- **Tiles**: Stage-specific routing circuits
- **Composition**: Cooley-Tukey structure
- **Verification**: 100% on all circuits

---

## Limitations and Future Work

### Current Limitations

1. **Scale**: Tested up to N=32; larger transforms need more twiddle opcodes
2. **Precision**: FP32 twiddles; could use FP16 or fixed-point
3. **Hardware**: Software prototype; no silicon validation

### Future Directions

1. **Other transforms**: DCT, NTT (Number Theoretic Transform)
2. **Larger N**: N=64, 128, 256 with hierarchical opcode tables
3. **Hardware synthesis**: FPGA/ASIC implementation of opcode dispatch
4. **Integration**: Connect to TriX tile routing system

---

## Code Availability

All code is in the TriX repository:

```
experiments/fft_atoms/fft_compiler.py  # Main implementation
docs/FFT_COMPILATION.md                # Technical documentation
docs/TWIDDLE_OPCODES.md                # Twiddle opcode details
docs/RESEARCH_SUMMARY.md               # This document
```

---

## Key Equations

### WHT Partner Selection
```
partner(stage, pos) = pos ⊕ 2^stage
```

### DFT Twiddle Index
```
k = j × (N / m)   where m = 2^(stage+1)
```

### Twiddle Opcode
```
(re, im) × W_N^k = (re·cos(2πk/N) - im·sin(2πk/N), 
                   re·sin(2πk/N) + im·cos(2πk/N))
```

### FP4 Threshold Circuit
```
y = σ(Σ w_i x_i + b)   where w_i ∈ {-1, 0, +1}, b ∈ {-2.5, ..., 1.5}
```

---

## Acknowledgments

- **VGem**: Twiddle opcode architecture and verification guards
- **Nova**: "Don't train atoms to be exact. Construct them to be exact."
- **TriX team**: Foundation architecture

---

## The Punchline

> "You didn't fail. You discovered what you built."
> — VGem, on the WHT/DFT revelation

> "No runtime math. Twiddles become opcodes. Routing selects them."
> — VGem, on the path to true compiled DFT

---

*Neural computation is not approximation. It can be compilation.*
