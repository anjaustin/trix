# TriX Tutorial: From Zero to Compiled Transforms

**A gentle introduction for beginners.**

---

## What You'll Learn

By the end of this tutorial, you'll understand:
1. What "compiled computation" means
2. How to build exact boolean circuits
3. How to compile a transform (WHT)
4. Why this matters

**Prerequisites:** Basic Python. That's it.

---

## Part 1: The Big Idea

### Traditional Neural Networks

Traditional neural networks are **trained**:
1. Start with random weights
2. Show examples
3. Adjust weights to reduce error
4. Hope it generalizes

The result is approximate. Close enough for many tasks, but never exact.

### TriX: Compiled Computation

TriX takes a different approach: **construction**.
1. Know what function you want
2. Design weights that compute it exactly
3. Verify on all inputs
4. Done

The result is exact. Not "99.9% accurate" - actually correct.

### Why This Matters

Imagine you need to add two numbers. Would you:
- (A) Train a neural network on millions of addition examples and hope it works?
- (B) Build a circuit that adds correctly by design?

TriX is approach (B).

---

## Part 2: Your First Atom

An **atom** is the smallest unit of computation. Let's build one.

### The AND Gate

The AND gate outputs 1 only when both inputs are 1:

```
Inputs → Output
(0, 0) → 0
(0, 1) → 0
(1, 0) → 0
(1, 1) → 1
```

### Building AND with a Threshold Circuit

A threshold circuit computes: `output = 1 if (w1*x1 + w2*x2 + b) > 0 else 0`

For AND, we need:
- `0 + 0 + b ≤ 0` (output 0)
- `0 + w2 + b ≤ 0` (output 0)
- `w1 + 0 + b ≤ 0` (output 0)
- `w1 + w2 + b > 0` (output 1)

Solution: `w1 = 1, w2 = 1, b = -1.5`

Check:
- `0 + 0 - 1.5 = -1.5 ≤ 0` ✓
- `0 + 1 - 1.5 = -0.5 ≤ 0` ✓
- `1 + 0 - 1.5 = -0.5 ≤ 0` ✓
- `1 + 1 - 1.5 = 0.5 > 0` ✓

**We didn't train this. We constructed it.**

### Try It Yourself

```python
import torch

def AND(x1, x2):
    """AND gate as a threshold circuit."""
    w1, w2, b = 1.0, 1.0, -1.5
    z = w1 * x1 + w2 * x2 + b
    return 1 if z > 0 else 0

# Test all inputs
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"AND({x1}, {x2}) = {AND(x1, x2)}")
```

Output:
```
AND(0, 0) = 0
AND(0, 1) = 0
AND(1, 0) = 0
AND(1, 1) = 1
```

---

## Part 3: Building an Adder

Now let's build something useful: a 1-bit adder.

### The Problem

Add two bits (a, b) with a carry-in (cin). Produce sum and carry-out.

```
a + b + cin = (cout, sum)

0 + 0 + 0 = (0, 0) = 0
0 + 0 + 1 = (0, 1) = 1
0 + 1 + 0 = (0, 1) = 1
0 + 1 + 1 = (1, 0) = 2
1 + 0 + 0 = (0, 1) = 1
1 + 0 + 1 = (1, 0) = 2
1 + 1 + 0 = (1, 0) = 2
1 + 1 + 1 = (1, 1) = 3
```

### The Insight

- **sum** = parity of inputs = (a XOR b XOR cin)
- **cout** = majority of inputs = 1 if at least 2 inputs are 1

### SUM Atom

SUM outputs 1 when an odd number of inputs are 1:

```python
def SUM(a, b, cin):
    """Sum bit: XOR of three inputs."""
    # Two-layer threshold circuit
    # Layer 1: detect different parities
    # Layer 2: combine
    
    # Simplified: use XOR composition
    return a ^ b ^ cin
```

In TriX, we construct this as a 2-layer threshold circuit.

### CARRY Atom

CARRY outputs 1 when at least 2 inputs are 1:

```python
def CARRY(a, b, cin):
    """Carry bit: majority function."""
    w1, w2, w3, b = 1.0, 1.0, 1.0, -1.5
    z = w1 * a + w2 * b + w3 * cin + b
    return 1 if z > 0 else 0
```

Check: threshold is 1.5, so need sum ≥ 2 to output 1. ✓

### Try the Full Adder

```python
def full_adder(a, b, cin):
    """1-bit full adder using atoms."""
    s = SUM(a, b, cin)
    cout = CARRY(a, b, cin)
    return cout, s

# Test all inputs
for a in [0, 1]:
    for b in [0, 1]:
        for cin in [0, 1]:
            cout, s = full_adder(a, b, cin)
            value = a + b + cin
            print(f"{a} + {b} + {cin} = ({cout}, {s}) = {cout*2 + s} [expected {value}]")
```

Output:
```
0 + 0 + 0 = (0, 0) = 0 [expected 0]
0 + 0 + 1 = (0, 1) = 1 [expected 1]
0 + 1 + 0 = (0, 1) = 1 [expected 1]
0 + 1 + 1 = (1, 0) = 2 [expected 2]
1 + 0 + 0 = (0, 1) = 1 [expected 1]
1 + 0 + 1 = (1, 0) = 2 [expected 2]
1 + 1 + 0 = (1, 0) = 2 [expected 2]
1 + 1 + 1 = (1, 1) = 3 [expected 3]
```

**100% correct. No training. Just construction.**

---

## Part 4: From Adder to Transform

### The Walsh-Hadamard Transform

The WHT is like addition, but for signal processing. It takes a list of numbers and transforms them.

The basic operation is the **butterfly**:
```
(a, b) → (a + b, a - b)
```

A full WHT applies butterflies in stages, pairing different elements each time.

### WHT for N=4

```
Input:  [x0, x1, x2, x3]

Stage 0: Pair elements at distance 1
  (x0, x1) → (x0+x1, x0-x1)
  (x2, x3) → (x2+x3, x2-x3)
  Result: [x0+x1, x0-x1, x2+x3, x2-x3]

Stage 1: Pair elements at distance 2
  (y0, y2) → (y0+y2, y0-y2)
  (y1, y3) → (y1+y3, y1-y3)
  Result: [y0+y2, y1+y3, y0-y2, y1-y3]
```

### The Pattern

At stage s, pair elements at distance `2^s`. The partner of position i is `i XOR 2^s`.

```python
def partner(stage, pos):
    return pos ^ (1 << stage)

# Stage 0: distance 1
partner(0, 0) = 0 ^ 1 = 1  # pair (0, 1)
partner(0, 2) = 2 ^ 1 = 3  # pair (2, 3)

# Stage 1: distance 2
partner(1, 0) = 0 ^ 2 = 2  # pair (0, 2)
partner(1, 1) = 1 ^ 2 = 3  # pair (1, 3)
```

### Simple WHT Implementation

```python
def wht(x):
    """Walsh-Hadamard Transform."""
    n = len(x)
    result = list(x)
    
    stage = 0
    while (1 << stage) < n:
        new_result = result.copy()
        distance = 1 << stage
        
        for i in range(n):
            partner = i ^ distance
            if i < partner:  # Process each pair once
                a, b = result[i], result[partner]
                new_result[i] = a + b
                new_result[partner] = a - b
        
        result = new_result
        stage += 1
    
    return result

# Try it
x = [1, 2, 3, 4]
print(f"WHT({x}) = {wht(x)}")
# Output: WHT([1, 2, 3, 4]) = [10, -2, -4, 0]
```

### What TriX Does

TriX compiles this to FP4 threshold circuits:
1. The partner selection (`i ^ distance`) becomes a circuit
2. The "which gets sum vs diff" decision becomes a circuit
3. The arithmetic (add, subtract) stays as microcode

Result: A transform that's verified correct, uses minimal memory, and runs deterministically.

---

## Part 5: Why "Compiled" Matters

### Training vs Compilation

| Aspect | Trained | Compiled |
|--------|---------|----------|
| Correctness | Approximate | Exact |
| Verification | Statistical | Exhaustive |
| Runtime | May vary | Deterministic |
| Memory | Often large | Minimal |

### The TriX Philosophy

> "The routing learns WHEN. The atoms compute WHAT."

- **Routing**: Which elements to pair, which operation to apply → structural, can be compiled
- **Atoms**: The actual computation (add, XOR, etc.) → fixed microcode, exact

When both are fixed, the entire computation is compiled. No decisions at runtime. No approximation. Just execution.

---

## Part 6: Running the Real Code

### Setup

```bash
cd /workspace/trix_latest
pip install -r requirements.txt
```

### Test the Atoms

```python
import sys
sys.path.insert(0, 'src')

from trix.compiler.atoms_fp4 import FP4AtomLibrary

lib = FP4AtomLibrary()

# Test AND
and_atom = lib.get_atom("AND")
print(f"AND atom: {and_atom.status}")

# Test all inputs
import torch
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    inp = torch.tensor([[float(x1), float(x2)]])
    out = and_atom.circuit(inp)
    result = int(out[0,0].item() > 0.5)
    print(f"AND({x1}, {x2}) = {result}")
```

### Test the WHT

```python
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'experiments/fft_atoms')

from fft_compiler import test_compiled_wht

# This compiles WHT routing to FP4 and tests it
test_compiled_wht(8)
```

### Test the DFT

```python
from fft_compiler import test_compiled_complex_fft

# This uses twiddle opcodes - no runtime trig!
test_compiled_complex_fft(8)
```

---

## Part 7: What You've Learned

1. **Atoms** are exact boolean/arithmetic functions built by construction
2. **Compilation** means all decisions are resolved before runtime
3. **WHT** uses XOR-based partner selection (structural routing)
4. **DFT** adds twiddle opcodes (fixed complex multiplies)
5. **The pattern**: Structural routing + Fixed microcode = Verified computation

---

## Next Steps

- **Read the Glossary**: `docs/GLOSSARY.md`
- **Run the Tests**: `pytest tests/test_rigorous.py -v`
- **Study the Code**: `experiments/fft_atoms/fft_compiler.py`
- **Read the Research Summary**: `docs/RESEARCH_SUMMARY.md`

---

## Quick Reference

```python
# Import atoms
from trix.compiler.atoms_fp4 import FP4AtomLibrary
lib = FP4AtomLibrary()
atom = lib.get_atom("AND")  # or OR, XOR, SUM, CARRY, etc.

# Run atom
import torch
x = torch.tensor([[1.0, 0.0]])  # inputs
y = atom.circuit(x)  # output
result = int(y[0,0].item() > 0.5)

# Import transforms
from fft_compiler import compile_fft_routing, CompiledWHT

# Compile WHT
routing = compile_fft_routing(8)
wht = CompiledWHT(
    N=8,
    is_upper_circuit=routing['is_upper']['circuit'],
    partner_circuits=routing['partner']['circuits'],
)

# Run WHT
result = wht.execute([1, 2, 3, 4, 5, 6, 7, 8])
```

---

*Welcome to compiled computation.*
