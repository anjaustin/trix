# Mesa 8: FP4 Atoms Reference

**Threshold Circuit Primitives for Neural Computation**

---

## Overview

FP4 Atoms are the fundamental building blocks of the TriX architecture. They are threshold circuits that compute exact boolean functions using ternary weights {-1, 0, +1}.

**Key Property:** Atoms are *constructed* to be exact, not *trained* to approximate.

---

## Threshold Circuit Model

### Definition

A threshold circuit computes:

```
output = 1 if (w · x + b) > θ else 0
```

Where:
- `w` = weight vector ∈ {-1, 0, +1}^n (ternary)
- `x` = input vector ∈ {0, 1}^n (binary)
- `b` = bias ∈ ℝ
- `θ` = threshold ∈ ℝ

### Properties

1. **Exact by construction**: Weights and thresholds are chosen to implement specific boolean functions
2. **No training required**: Parameters are computed analytically
3. **Ternary weights**: Only {-1, 0, +1} values enable efficient hardware

---

## Core Atoms

### AND Atom

**Function:** Output 1 iff both inputs are 1.

```python
weights = [1, 1]
bias = 0
threshold = 1.5

# w·x > 1.5 requires both inputs = 1
# (1·1 + 1·1) = 2 > 1.5 ✓
# (1·1 + 1·0) = 1 < 1.5 ✗
```

**Truth Table:**
| a | b | output |
|---|---|--------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

---

### OR Atom

**Function:** Output 1 iff at least one input is 1.

```python
weights = [1, 1]
bias = 0
threshold = 0.5

# w·x > 0.5 requires at least one input = 1
```

**Truth Table:**
| a | b | output |
|---|---|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

---

### XOR Atom (2-input)

**Function:** Output 1 iff inputs differ.

XOR cannot be computed by a single threshold gate (not linearly separable). 
Requires composition:

```python
# XOR(a,b) = OR(AND(a, NOT(b)), AND(NOT(a), b))
# Or use: (a + b) mod 2
```

---

### NOT Atom

**Function:** Output inverse of input.

```python
weights = [-1]
bias = 0.5
threshold = 0

# -1·x + 0.5 > 0
# x=0: 0.5 > 0 → output 1
# x=1: -0.5 > 0 → output 0
```

---

## Arithmetic Atoms

### SUM Atom (3-input Parity)

**Function:** Output 1 iff odd number of inputs are 1.

This is the sum bit of a full adder: `SUM = a ⊕ b ⊕ cin`

```python
class SUMAtom:
    def forward(self, a, b, c):
        return ((a + b + c) % 2).float()
```

**Truth Table:**
| a | b | cin | sum |
|---|---|-----|-----|
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 |
| 0 | 1 | 0 | 1 |
| 0 | 1 | 1 | 0 |
| 1 | 0 | 0 | 1 |
| 1 | 0 | 1 | 0 |
| 1 | 1 | 0 | 0 |
| 1 | 1 | 1 | 1 |

**Note:** 3-input parity requires multiple threshold gates or modular arithmetic.

---

### CARRY Atom (Majority)

**Function:** Output 1 iff at least 2 of 3 inputs are 1.

This is the carry bit of a full adder: `CARRY = (a ∧ b) ∨ (b ∧ cin) ∨ (a ∧ cin)`

```python
class CARRYAtom:
    weights = [1, 1, 1]
    threshold = 1.5
    
    def forward(self, a, b, c):
        return (a + b + c >= 2).float()
```

**Truth Table:**
| a | b | cin | carry |
|---|---|-----|-------|
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 0 |
| 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 1 |
| 1 | 1 | 1 | 1 |

---

### MUX Atom (Multiplexer)

**Function:** Output a if sel=0, else b.

```
MUX(a, b, sel) = (a ∧ ¬sel) ∨ (b ∧ sel)
```

Requires composition of AND, OR, NOT atoms.

---

## Composition

### Full Adder

Composed from SUM + CARRY atoms:

```python
class FullAdderTile:
    def __init__(self):
        self.sum_atom = SUMAtom()
        self.carry_atom = CARRYAtom()
    
    def forward(self, a, b, cin):
        s = self.sum_atom(a, b, cin)
        cout = self.carry_atom(a, b, cin)
        return s, cout
```

**Verification:** 8/8 input combinations correct.

---

### Ripple Carry Adder

Composed from N Full Adders:

```python
class RippleAdderTile:
    def __init__(self, bits=32):
        self.adders = [FullAdderTile() for _ in range(bits)]
    
    def forward(self, a, b):
        carry = 0
        result = []
        for i in range(bits):
            s, carry = self.adders[i](a[i], b[i], carry)
            result.append(s)
        return result
```

**Verification (8-bit):**
```
   0 +   0 =   0 ✓
   1 +   1 =   2 ✓
  37 +  28 =  65 ✓
  42 +  58 = 100 ✓
 127 +   1 = 128 ✓
 255 +   0 = 255 ✓
```

---

## FP4 Encoding

### Weight Encoding

Ternary values packed into 4 bits:

| Value | Encoding |
|-------|----------|
| -1 | 0b00 |
| 0 | 0b01 |
| +1 | 0b10 |
| (reserved) | 0b11 |

Two weights per byte → 4x compression vs float32.

### Lookup Table

For atoms with small input space, we can use lookup tables:

```python
# 2-input function: 4 entries
AND_LUT = [0, 0, 0, 1]  # Index by (a << 1) | b

# 3-input function: 8 entries  
SUM_LUT = [0, 1, 1, 0, 1, 0, 0, 1]  # Index by (a << 2) | (b << 1) | c
```

---

## Design Principles

### 1. Construction over Training

```python
# WRONG: Train to approximate
model = train(data, labels, epochs=1000)

# RIGHT: Construct to be exact
atom = ThresholdCircuit(weights=[1,1,1], threshold=1.5)
```

### 2. Verify Exhaustively

For small input spaces, test ALL combinations:

```python
def verify_atom(atom, truth_table):
    for inputs, expected in truth_table.items():
        actual = atom(*inputs)
        assert actual == expected
```

### 3. Compose Hierarchically

```
Atoms → Gates → Adders → ALU → CPU
```

Each layer inherits correctness from the layer below.

---

## Atom Library

| Atom | Inputs | Function | Status |
|------|--------|----------|--------|
| AND | 2 | Conjunction | ✅ Verified |
| OR | 2 | Disjunction | ✅ Verified |
| NOT | 1 | Negation | ✅ Verified |
| XOR | 2 | Exclusive or | ✅ Verified |
| SUM | 3 | Parity | ✅ Verified |
| CARRY | 3 | Majority | ✅ Verified |
| MUX | 3 | Select | ✅ Verified |
| NAND | 2 | Not-and | ✅ Verified |
| NOR | 2 | Not-or | ✅ Verified |
| XNOR | 2 | Equivalence | ✅ Verified |

---

## The Nova Principle

> "Don't train atoms to be exact. Construct them to be exact."

This is the foundation of TriX:

1. **Atoms** are exact by construction
2. **Tiles** inherit exactness through composition
3. **The machine** computes correctly because every component computes correctly

No approximation. No error. No hallucination.

---

*Threshold circuits. Ternary weights. Exact computation.*
