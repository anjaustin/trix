# Tesseract 6502: Atomic Composition

> **⚠️ EXPERIMENTAL** - This documents speculative research from the Riemann Zero Hunter session. The stable TriX release (v0.9.x) does not include this work.

## The Discovery

The 6502 microprocessor operations can be perfectly emulated using **atomic composition** rather than learned approximation.

### Key Insight

> **"ATOMIZE: Tiles ARE atoms. Routing IS composition."**

Instead of training a neural network to approximate ADC (add with carry), we:
1. Build exact atomic operations (ADD, INC, AND, OR, XOR, shifts)
2. Route to atoms via XOR signature matching
3. Compose atoms for complex operations (ADC = ADD → INC when carry set)

### Results

| Approach | ADC Accuracy | Overall |
|----------|--------------|---------|
| Vanilla TriX (learned) | 4.8% | 87.6% |
| GeometriX (spatial) | 30.6% | 88.2% |
| **Atomic Composition** | **100%** | **100%** |

**589,824 test cases. Zero errors.**

## The Tesseract Structure

The 6502 maps to a 29-dimensional hypercube:

```
Dimension Allocation:
  Bits 0-7:   Opcode (256 operations)
  Bits 8-15:  Operand A (256 values)
  Bits 16-23: Operand B (256 values)  
  Bit 24:     Carry flag
  Bits 25-28: Operation type (16 types)

Total: 29 dimensions = 537M vertices
```

### XOR Navigation

Moving between tesseract vertices uses XOR:

```python
# Current state
vertex_a = 0b00000000000000000000000000001  # ADC with C=0

# Flip carry dimension
vertex_b = vertex_a ^ (1 << 24)              # ADC with C=1

# Distance = popcount(vertex_a ^ vertex_b) = 1 hop
```

## Atomic Operations

### Exact Atoms (Frozen, 100% Accurate)

| Atom | Operation | Implementation |
|------|-----------|----------------|
| ADD | a + b (no carry) | `(a + b) & 0xFF` |
| INC | a + 1 | `(a + 1) & 0xFF` |
| DEC | a - 1 | `(a - 1) & 0xFF` |
| AND | a & b | Bitwise AND |
| ORA | a \| b | Bitwise OR |
| EOR | a ^ b | Bitwise XOR |
| ASL | a << 1 | Arithmetic shift left |
| LSR | a >> 1 | Logical shift right |

### Composition Rules

| Operation | Composition |
|-----------|-------------|
| ADC (C=0) | ADD |
| ADC (C=1) | ADD → INC |
| SBC (C=1) | ADD (with complement) |
| SBC (C=0) | ADD → DEC |

## Implementation

### XOR Routing

```python
def route_to_atom(opcode: int, carry: bool) -> Atom:
    """Route via XOR signature matching."""
    signature = opcode | (carry << 8)
    
    # XOR distance to each atom signature
    distances = [popcount(signature ^ atom.sig) for atom in ATOMS]
    
    # Return nearest atom
    return ATOMS[argmin(distances)]
```

### Atomic Execution

```python
def execute_6502_op(opcode, a, b, carry):
    """Execute via atomic composition."""
    if opcode == ADC:
        result = ADD_ATOM(a, b)
        if carry:
            result = INC_ATOM(result)
        return result
    # ... other operations
```

## Files

| File | Purpose |
|------|---------|
| `experiments/trix_6502_atomic_simple.py` | Simple composition proof |
| `experiments/trix_6502_full_stack.py` | Complete implementation |
| `experiments/trix_6502_comparison.py` | Comparison with learned approaches |
| `src/trix/nn/xor_routing.py` | XOR signature routing |

## Why This Matters

### For Deterministic Systems

Neural networks approximate. Atoms compute exactly.

For any deterministic operation (CPU instructions, cryptography, error correction), atomic composition achieves:
- **100% accuracy** (not 99.9%)
- **Zero training** (atoms are frozen)
- **O(1) routing** (XOR is fast)

### Tesseract Strengths

| Domain | Why Tesseract Excels |
|--------|---------------------|
| 6502 emulation | All discrete ops, XOR routing |
| Binary/Ternary NNs | Sign detection at 3.76B/sec |
| State machines | XOR navigation between states |
| Error correction | Hamming distance = XOR + popcount |

### Tesseract Weaknesses

| Domain | Why Tesseract Struggles |
|--------|------------------------|
| Riemann zeros | Requires cos(), log() transcendentals |
| Continuous math | FP32/FP64 bottlenecked |
| Scaling sums | O(√t) terms unavoidable |

## The Lesson

> "Don't learn what you can compute."

The 6502 was the proof. 589,824 test cases, 0 errors. Because every operation is **discrete and deterministic**.

Riemann needs cos(). The 6502 doesn't.

**Match the tool to the problem.**
