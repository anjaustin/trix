# Transform Compilation to FP4

**Compile transform routing to FP4 threshold circuits.**

---

## Important Distinction

We implement **two different transforms**:

| Transform | Description | Routing |
|-----------|-------------|---------|
| **Walsh-Hadamard (WHT)** | Real-valued, self-inverse | FP4 compiled circuits |
| **Discrete Fourier (DFT/FFT)** | Complex-valued, Cooley-Tukey | Standard algorithm |

The XOR-based pairing (partner = pos XOR 2^stage) gives **WHT**, not DFT!

---

## Overview

Transforms have two types of operations:

| Type | Examples | Implementation |
|------|----------|-------------|
| **Structural** | Partner index, twiddle selection | FP4 threshold circuits |
| **Arithmetic** | a+b, a-b, complex multiply | Fixed microcode (exact) |

We compile the ROUTING to FP4. The arithmetic stays exact.

---

## Quick Start

```python
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'experiments/fft_atoms')

from fft_compiler import test_compiled_wht, test_compiled_complex_fft

# Walsh-Hadamard Transform (compiled routing)
test_compiled_wht(8)   # 100% exact
test_compiled_wht(16)  # 100% exact
test_compiled_wht(32)  # 100% exact

# Complex FFT (Cooley-Tukey DIT)
test_compiled_complex_fft(8)   # Exact to float precision
test_compiled_complex_fft(16)  # Exact to float precision
```

---

## Architecture

### Walsh-Hadamard Transform (WHT)

XOR-based structure:
- `log2(N)` stages
- Partner of position `i` at stage `s`: `i XOR 2^s`
- Butterfly: (a+b, a-b) - no twiddles!
- Self-inverse: WHT(WHT(x)) = N * x

### Discrete Fourier Transform (DFT)

Cooley-Tukey DIT structure:
- `log2(N)` stages with groups of size `2^(s+1)`
- Requires bit-reversed input
- Butterfly: (u + W*t, u - W*t) with twiddle factors

### Compiled Components (WHT)

```
┌─────────────────────────────────────────────────────────────────┐
│                         COMPILED WHT                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IS_UPPER(stage, pos) → 0 or 1                                 │
│    Determines: does this position get SUM or DIFF?             │
│    Implementation: FP4 threshold circuit                        │
│    Verification: 100%                                           │
│                                                                 │
│  PARTNER(stage, pos) → partner_index                           │
│    Determines: which element to pair with?                      │
│    Implementation: FP4 threshold circuits (1 per output bit)   │
│    Verification: 100%                                           │
│                                                                 │
│  BUTTERFLY(a, b) → (a+b, a-b)                                  │
│    Fixed microcode - exact arithmetic                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why WHT Routing Compiles

The key insight: WHT routing is **structural**, not value-dependent.

Given (stage, position), we can compute:
- Partner index: `pos XOR 2^stage`
- Is upper: `pos > partner`

These are **discrete functions** on bounded inputs. They can be expressed as boolean circuits and compiled to FP4 threshold circuits.

---

## API Reference

### compile_fft_routing(N)

Compile routing for real FFT.

```python
from fft_compiler import compile_fft_routing

routing = compile_fft_routing(8)

# Returns dict with:
# - routing['is_upper']['circuit']   - ThresholdCircuit
# - routing['is_upper']['verified']  - bool
# - routing['partner']['circuits']   - List of ThresholdCircuit
# - routing['partner']['verified']   - bool
```

### compile_complex_fft_routing(N)

Compile routing including twiddle selection.

```python
from fft_compiler import compile_complex_fft_routing

routing = compile_complex_fft_routing(8)

# Additional:
# - routing['twiddle']['circuits']   - List of ThresholdCircuit
# - routing['twiddle']['verified']   - bool
```

### CompiledWHT

Execute Walsh-Hadamard Transform with compiled routing.

```python
from fft_compiler import CompiledWHT, compile_fft_routing

routing = compile_fft_routing(8)
wht = CompiledWHT(
    N=8,
    is_upper_circuit=routing['is_upper']['circuit'],
    partner_circuits=routing['partner']['circuits'],
)

# Execute
result = wht.execute([1, 2, 3, 4, 5, 6, 7, 8])
# Result: [36, -4, -8, 0, -16, 0, 0, 0]

# Self-inverse property
inverse = wht.execute(result)
# inverse/N = original
```

### CompiledComplexFFT

Execute complex FFT with compiled routing.

```python
from fft_compiler import CompiledComplexFFT, compile_complex_fft_routing

routing = compile_complex_fft_routing(8)
fft = CompiledComplexFFT(
    N=8,
    is_upper_circuit=routing['is_upper']['circuit'],
    partner_circuits=routing['partner']['circuits'],
    twiddle_circuits=routing['twiddle']['circuits'],
)

# Execute (real input)
real_out, imag_out = fft.execute([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

# Execute (complex input)
real_out, imag_out = fft.execute(real_list, imag_list)
```

---

## Results

### Walsh-Hadamard Transform (WHT)

| N | IS_UPPER | PARTNER | WHT Accuracy |
|---|----------|---------|--------------|
| 8 | 100% (2 layers) | 100% (2 layers/bit) | **100% exact** |
| 16 | 100% (2 layers) | 100% (2 layers/bit) | **100% exact** |
| 32 | 100% (2 layers) | 100% (2 layers/bit) | **100% exact** |

### Complex FFT (DFT) - TRUE COMPILED

| N | Accuracy vs NumPy | Twiddle Opcodes |
|---|-------------------|-----------------|
| 8 | **0.00** (exact!) | 4 opcodes used |
| 16 | **~2e-15** (float precision) | 8 opcodes used |

**Key achievement:** NO RUNTIME TRIG! Twiddles are fixed microcode opcodes.

The complex FFT uses:
- Cooley-Tukey DIT algorithm
- Bit-reversed input
- **Twiddle opcodes** (not `np.cos/sin`)
- Structural routing: `tw_idx = j * (N // m)`

---

## How WHT Compilation Works

### Step 1: Analyze Structure

```python
def analyze_fft_structure(N):
    # For each (stage, pos), compute:
    # - partner = pos XOR 2^stage
    # - is_upper = 1 if pos > partner else 0
    ...
```

### Step 2: Build Truth Tables

```python
def build_is_upper_truth_table(N):
    # Input: (stage_bits, pos_bits)
    # Output: (is_upper_bit)
    # Example for N=8:
    #   (0, 0, 0, 0, 0) → 0  (stage=0, pos=0, partner=1, 0<1)
    #   (0, 0, 0, 0, 1) → 1  (stage=0, pos=1, partner=0, 1>0)
    ...
```

### Step 3: Compile to Threshold Circuits

```python
from trix.compiler.atoms_fp4 import truth_table_to_circuit

circuit = truth_table_to_circuit("IS_UPPER", num_inputs, truth_table)
# Returns ThresholdCircuit with FP4-compatible weights
```

### Step 4: Execute with Compiled Routing

```python
def execute(self, x):
    for stage in range(num_stages):
        for pos in range(N):
            partner = self._get_partner(stage, pos)  # Compiled circuit
            is_upper = self._get_is_upper(stage, pos)  # Compiled circuit
            
            # Exact arithmetic
            sum_val = x[pos] + x[partner]
            diff_val = x[pos] - x[partner]
            
            # Assign based on compiled is_upper
            ...
```

## How True Compiled DFT Works

### Twiddle Opcodes (The Key Innovation)

**Before (runtime trig):**
```python
wm_re = np.cos(-2 * np.pi / m)  # BAD: computed at runtime
wm_im = np.sin(-2 * np.pi / m)
```

**After (fixed microcode):**
```python
# Twiddle opcodes - fixed constants, no runtime trig!
TWIDDLE_TABLE_8 = [
    ( 1.0,         0.0),         # k=0: W^0 = 1
    ( SQRT_HALF,  -SQRT_HALF),   # k=1: W^1
    ( 0.0,        -1.0),         # k=2: W^2 = -i
    (-SQRT_HALF,  -SQRT_HALF),   # k=3: W^3
    (-1.0,         0.0),         # k=4: W^4 = -1
    (-SQRT_HALF,   SQRT_HALF),   # k=5: W^5
    ( 0.0,         1.0),         # k=6: W^6 = i
    ( SQRT_HALF,   SQRT_HALF),   # k=7: W^7
]

# Opcode: fixed complex multiply
def twiddle_op(re, im) -> (re*c - im*s, re*s + im*c)
```

### Structural Routing to Opcodes

```python
# Twiddle index is STRUCTURAL - no learning, no runtime computation
tw_idx = (j * (N // m)) % N

# Execute via opcode, not computation
wt_re, wt_im = twiddle_ops[tw_idx](t_re, t_im)
```

### The Complete Compiled DFT

```python
def execute(self, x_real, x_imag):
    # Bit-reverse inputs
    values_re = self._bit_reverse_array(x_real)
    values_im = self._bit_reverse_array(x_imag)
    
    for stage in range(num_stages):
        m = 2 ** (stage + 1)
        
        for k in range(0, N, m):
            for j in range(m // 2):
                idx_u, idx_t = k + j, k + j + m//2
                
                # TWIDDLE OPCODE: structural routing to fixed microcode
                tw_idx = get_twiddle_index(N, m, j)  # Structural!
                wt_re, wt_im = twiddle_ops[tw_idx](t_re, t_im)  # Microcode!
                
                # Butterfly
                new[idx_u] = u + wt
                new[idx_t] = u - wt
```

**No `np.cos`, `np.sin`, or `exp` anywhere in the execution path!**

---

## Theory

### Transforms as Routing + Microcode

Traditional transform implementations hardcode the routing (partner selection, twiddle indices). Our approach:

1. **Extract** the routing logic as truth tables
2. **Compile** to FP4 threshold circuits
3. **Execute** with compiled routing + exact arithmetic

This separates:
- **WHAT** to compute (exact butterfly operations)
- **WHEN/WHERE** to route (compiled FP4 circuits)

### Bit-Level Encoding

For N=8 WHT:
- Stage: 2 bits (0-2)
- Position: 3 bits (0-7)
- Total input: 5 bits

The IS_UPPER circuit takes 5 bits, outputs 1 bit.
The PARTNER circuits take 5 bits, output 3 bits (one circuit per bit).

### WHT vs DFT

| Property | WHT | DFT |
|----------|-----|-----|
| Domain | Real | Complex |
| Twiddles | None | e^(-2*pi*i*k/N) |
| Self-inverse | Yes (WHT(WHT(x)) = Nx) | No (requires IFFT) |
| Compiled routing | Yes | Possible but complex |
| Applications | Compression, quantum | Signal processing |

---

## Files

| File | Purpose |
|------|---------|
| `experiments/fft_atoms/fft_compiler.py` | Transform compilation |
| `docs/FFT_COMPILATION.md` | This document |
| `notes/ROADMAP_FP4.md` | Development roadmap |

---

## Relation to TriX

Transform compilation demonstrates the TriX pattern at algorithm level:

| TriX Concept | Transform Analog |
|--------------|-----------------|
| Tiles | Butterfly operations |
| Routing | Partner selection |
| Signatures | (stage, pos) encoding |
| Verification | 100% on all circuits |

The same "compiled control + exact computation" pattern applies.

---

## Future Work

1. Compile DFT routing to FP4 (different structure than WHT)
2. Extend to N=64, 128, 256
3. Emit compiled transforms as .trix.json + .fp4 files
4. Build transform-specific inference kernels
5. Explore other transforms (DCT, NTT)

---

*Transforms: Where routing becomes the algorithm.*
