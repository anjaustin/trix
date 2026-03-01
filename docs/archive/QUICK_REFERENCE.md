# TriX Quick Reference

**One-page summary for researchers.**

---

## The Pattern

```
Structural Routing + Fixed Microcode = Verified Computation
```

---

## What We Built

| Component | Method | Accuracy |
|-----------|--------|----------|
| FP4 Threshold Atoms | Construction (not training) | 100% |
| WHT Routing | FP4 circuit compilation | 100% |
| DFT Twiddles | Fixed microcode opcodes | 0.00 error |

---

## Key Files

```
src/trix/compiler/
├── atoms.py          # Float atom library
├── atoms_fp4.py      # FP4 threshold circuits (construction)
├── fp4_pack.py       # 4-bit packing
├── emit.py           # Code generation
└── compiler.py       # Main compiler

experiments/fft_atoms/
└── fft_compiler.py   # Transform compilation

docs/
├── FFT_COMPILATION.md    # Transform overview
├── TWIDDLE_OPCODES.md    # Twiddle opcode details
├── FP4_INTEGRATION.md    # FP4 guide
└── RESEARCH_SUMMARY.md   # Research overview
```

---

## Key Equations

### FP4 Threshold Circuit
```
y = σ(Σ w_i x_i + b)
w_i ∈ {-1, 0, +1}
b ∈ {-2.5, -1.5, -0.5, 0.5, 1.5}
```

### WHT Partner
```
partner(stage, pos) = pos ⊕ 2^stage
```

### DFT Twiddle Index
```
k = j × (N / m)   where m = 2^(stage+1)
```

---

## Quick Start

### Run WHT Test
```python
from fft_compiler import test_compiled_wht
test_compiled_wht(8)   # 100% exact
```

### Run DFT Test
```python
from fft_compiler import test_compiled_complex_fft
test_compiled_complex_fft(8)  # 0.00 error
```

### Compile a Circuit
```python
from trix.compiler import TriXCompiler

compiler = TriXCompiler(use_fp4=True)
result = compiler.compile("full_adder")
# 100% exact, 58 bytes
```

---

## Key Insights

### From Nova
> "Don't train atoms to be exact. Construct them to be exact."

### From VGem
> "No runtime math. Twiddles become opcodes. Routing selects them."

### The Discovery
```
XOR pairing → WHT (not DFT!)
```
This wasn't a bug. It was understanding what the structure computes.

---

## Results Summary

| Test | Result |
|------|--------|
| FP4 Atoms (10 types) | 100% exact |
| Full Adder | 100% exact |
| 8-bit Adder | 100% exact |
| WHT N=8,16,32 | 100% exact |
| DFT N=8 | **0.00 error** |
| DFT N=16 | ~2e-15 |

---

## The Thesis

```
Neural computation is not approximation.
It can be compilation.
```

---

## Citation

```
TriX: Compiled Neural Computation
- FP4 Threshold Circuits: Exact by construction
- Transform Compilation: WHT + DFT via structural routing
- Pattern: Routing learns WHEN. Atoms compute WHAT.
```
