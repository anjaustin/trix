# TriX Tutorial: From Zero to Compiled Neural Computation

**A progressive introduction for beginners.**

---

## What is TriX?

TriX is a system for **compiled neural computation**. Instead of training neural networks and hoping they learn the right function, TriX:

1. **Constructs** exact atoms (tiny neural networks that compute boolean functions perfectly)
2. **Composes** them into larger circuits
3. **Verifies** correctness exhaustively
4. **Compiles** everything to efficient representations

The result: neural networks that compute **exactly**, with mathematical guarantees.

---

## Prerequisites

- Python 3.10+
- Basic understanding of boolean logic (AND, OR, XOR)
- Familiarity with NumPy arrays
- No deep learning expertise required!

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/trix.git
cd trix

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest tests/ -v --tb=short
```

---

## Part 1: Understanding Atoms

### What's an Atom?

An **atom** is the smallest unit of computation in TriX. It's a tiny neural network (typically 1-2 layers) that computes a specific boolean function **exactly**.

### Your First Atom: AND Gate

```python
import sys
sys.path.insert(0, 'src')

from trix.compiler.atoms_fp4 import FP4AtomLibrary
import torch

# Get the atom library
lib = FP4AtomLibrary()

# Get the AND atom
and_atom = lib.get_atom("AND")

# Test it on all inputs
print("AND Gate Truth Table:")
for a in [0, 1]:
    for b in [0, 1]:
        x = torch.tensor([[float(a), float(b)]])
        y = and_atom(x)
        result = int(y[0, 0].item() > 0.5)
        print(f"  AND({a}, {b}) = {result}")
```

Output:
```
AND Gate Truth Table:
  AND(0, 0) = 0
  AND(0, 1) = 0
  AND(1, 0) = 0
  AND(1, 1) = 1
```

### Available Atoms

| Atom | Inputs | Function |
|------|--------|----------|
| AND | 2 | Output 1 iff both inputs are 1 |
| OR | 2 | Output 1 iff at least one input is 1 |
| XOR | 2 | Output 1 iff inputs differ |
| NOT | 1 | Output inverse of input |
| SUM | 3 | Output parity of 3 inputs (a ⊕ b ⊕ c) |
| CARRY | 3 | Output 1 iff at least 2 inputs are 1 |
| MUX | 3 | Output a if sel=0, else b |

---

## Part 2: Composing Atoms

### Building a Full Adder

A full adder adds three bits (a, b, carry_in) and produces a sum and carry_out.

```python
# Full adder from SUM and CARRY atoms
sum_atom = lib.get_atom("SUM")
carry_atom = lib.get_atom("CARRY")

print("Full Adder Truth Table:")
print("  a  b  cin | sum cout")
print("  ----------|----------")

for a in [0, 1]:
    for b in [0, 1]:
        for cin in [0, 1]:
            x = torch.tensor([[float(a), float(b), float(cin)]])
            s = int(sum_atom(x)[0, 0].item() > 0.5)
            cout = int(carry_atom(x)[0, 0].item() > 0.5)
            print(f"  {a}  {b}  {cin}   |  {s}    {cout}")
```

### Chaining: 4-Bit Adder

```python
def add_4bit(a: int, b: int) -> int:
    """Add two 4-bit numbers using neural atoms."""
    # Extract bits (LSB first)
    a_bits = [(a >> i) & 1 for i in range(4)]
    b_bits = [(b >> i) & 1 for i in range(4)]
    
    carry = 0
    result_bits = []
    
    for i in range(4):
        x = torch.tensor([[float(a_bits[i]), float(b_bits[i]), float(carry)]])
        s = int(sum_atom(x)[0, 0].item() > 0.5)
        carry = int(carry_atom(x)[0, 0].item() > 0.5)
        result_bits.append(s)
    
    result_bits.append(carry)  # 5th bit for overflow
    
    # Convert back to integer
    return sum(bit << i for i, bit in enumerate(result_bits))

# Test
print("\n4-Bit Addition:")
print(f"  7 + 5 = {add_4bit(7, 5)}")
print(f"  15 + 1 = {add_4bit(15, 1)}")
print(f"  12 + 9 = {add_4bit(12, 9)}")
```

---

## Part 3: The Compiler

### Using the TriX Compiler

The compiler automates atom selection, wiring, and verification:

```python
from trix.compiler import TriXCompiler

compiler = TriXCompiler(use_fp4=True)

# Compile an 8-bit adder
result = compiler.compile("adder_8bit")

print(f"Compiled: {result.name}")
print(f"Atoms used: {result.stats['total_atoms']}")
print(f"Verified: {result.verified}")
```

### What the Compiler Does

1. **Specification**: Parses circuit description
2. **Decomposition**: Breaks into atoms and wiring
3. **Verification**: Tests all input combinations
4. **Emission**: Outputs optimized representation

---

## Part 4: Transform Compilation

### Compiling the Walsh-Hadamard Transform

TriX can compile signal processing transforms:

```python
sys.path.insert(0, 'experiments/fft_atoms')
from fft_compiler import compile_fft_routing, CompiledWHT

# Compile WHT for N=8
N = 8
routing = compile_fft_routing(N)

wht = CompiledWHT(
    N=N,
    is_upper_circuit=routing['is_upper']['circuit'],
    partner_circuits=routing['partner']['circuits'],
)

# Test it
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = wht.execute(x)

print(f"Input:  {x}")
print(f"Output: {y}")

# Verify against scipy
from scipy.linalg import hadamard
import numpy as np

H = hadamard(N)
expected = H @ np.array(x)
print(f"Expected: {expected.tolist()}")
print(f"Match: {np.allclose(y, expected)}")
```

### How It Works

1. **Routing circuits** determine which elements to pair
2. **Butterfly operations** apply 2×2 transforms to pairs
3. **Stages** repeat log₂(N) times

The routing is compiled to FP4 neural circuits!

---

## Part 5: Butterfly MatMul

### Structured Matrix Multiplication

The same structure that computes FFT can compute matrix operations:

```python
sys.path.insert(0, 'experiments/matmul')
from butterfly_matmul import identity_butterfly, hadamard_butterfly

# Identity matrix via butterfly
N = 8
net = identity_butterfly(N)
M = net.as_matrix()

print(f"Identity matrix error: {np.max(np.abs(M - np.eye(N)))}")

# Hadamard matrix via butterfly
net = hadamard_butterfly(N)
M = net.as_matrix()

from scipy.linalg import hadamard
print(f"Hadamard matrix error: {np.max(np.abs(M - hadamard(N)))}")
```

### The Unified Pattern

```
FFT:    Route → Twiddle → Route → Twiddle → ...
MatMul: Route → Block   → Route → Block   → ...
Both:   Route → Local   → Route → Local   → ...
```

---

## Part 6: The Isomorphic Transformer

### Replacing the Entire Transformer

TriX can replace both attention and MLP with structured operations:

```python
sys.path.insert(0, 'experiments/isomorphic')
from isomorphic_transformer import IsomorphicTransformer

# Create model
model = IsomorphicTransformer(
    vocab_size=100,
    seq_len=8,
    d_model=16,
    n_layers=2,
)

# Forward pass
import torch
x = torch.randint(0, 100, (2, 8))  # batch=2, seq_len=8
logits = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {logits.shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### What Makes It "Isomorphic"?

- **Attention** replaced by WHT/FFT (spectral mixing)
- **FFN** replaced by Butterfly MLP (structured matmul)
- **No O(N²) operations anywhere**
- Same structure, different instantiations

---

## Summary: The TriX Stack

```
Level 6: Isomorphic Transformer
         ↓
Level 5: Butterfly MatMul
         ↓
Level 4: FFT/WHT Compilation
         ↓
Level 3: TriX Compiler
         ↓
Level 2: Atom Composition
         ↓
Level 1: FP4 Atoms (threshold circuits)
```

Each level builds on the one below. All verified. All exact.

---

## Running the Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_rigorous.py -v

# Run with coverage
python -m pytest tests/ --cov=src/trix
```

---

## Next Steps

1. Read `docs/GLOSSARY.md` for terminology
2. Explore `experiments/` for advanced examples
3. Read `docs/FFT_COMPILATION.md` for transform details
4. Check `CHANGELOG.md` for recent developments

---

*Compiled neural computation. Exact results. Verified correctness.*
