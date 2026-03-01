# FP4 Integration Guide

**Complete guide to using FP4 atoms with the TriX Compiler.**

---

## Overview

The TriX Compiler supports two atom backends:

| Backend | Weights | Training | Accuracy |
|---------|---------|----------|----------|
| Float32 | `AtomLibrary` | Neural network training | 100% (trained) |
| FP4 | `FP4AtomLibrary` | Threshold circuits | 100% (by construction) |

FP4 atoms are **exact by construction** - no training convergence required.

---

## Quick Start

```python
from trix.compiler import TriXCompiler

# Create FP4 compiler
compiler = TriXCompiler(use_fp4=True)

# Compile a circuit
result = compiler.compile("adder_8bit")

# Execute
inputs = {"Cin": 0}
for i in range(8):
    inputs[f"A[{i}]"] = (37 >> i) & 1
    inputs[f"B[{i}]"] = (28 >> i) & 1

outputs = result.execute(inputs)
# Sum = 65, Cout = 0
```

---

## Architecture

### The FP4 Insight

> "Don't train atoms to be exact. Construct them to be exact."

FP4 atoms use **threshold circuits** - neural networks with hand-crafted weights that implement exact boolean logic.

```
AND(a,b) = step(a + b - 1.5)
OR(a,b)  = step(a + b - 0.5)
XOR(a,b) = step(step(a-b-0.5) + step(b-a-0.5) - 0.5)
```

All weights are in {-1, 0, +1}. All biases are in {-2.5, -1.5, -0.5, 0.5, 1.5}.

These values fit in FP4 formats (E2M1, NF4).

### Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     FP4 COMPILATION PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CircuitSpec                                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                               │
│  │  Decompose  │  Identify atom types needed                   │
│  └─────────────┘                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                               │
│  │   Verify    │  FP4 atoms are pre-verified (no training)     │
│  └─────────────┘                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                               │
│  │   Compose   │  Wire atoms into topology                     │
│  └─────────────┘                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                               │
│  │    Emit     │  Pack weights to 4-bit format                 │
│  └─────────────┘                                               │
│       │                                                         │
│       ▼                                                         │
│  .trix.json + weights/*.fp4                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## API Reference

### TriXCompiler

```python
compiler = TriXCompiler(
    use_fp4=True,      # Use FP4 threshold circuits
    verbose=True,      # Print progress
    cache_dir=None,    # Not used for FP4
)

result = compiler.compile(
    "adder_8bit",                    # Template name or CircuitSpec
    output_dir=Path("./output"),     # Emit files here
)
```

### CompilationResult

```python
result.success          # bool - compilation succeeded
result.spec             # CircuitSpec
result.topology         # Topology with tiles and routes
result.execute(inputs)  # Execute the circuit
result.summary()        # Human-readable summary
```

### FP4AtomLibrary

```python
from trix.compiler import FP4AtomLibrary

library = FP4AtomLibrary()

# Get an atom
atom = library.get_atom("SUM")

# Execute
import torch
x = torch.tensor([[1, 0, 1]], dtype=torch.float32)
y = atom(x)  # Output: [[0.0]] (1 XOR 0 XOR 1 = 0)

# List available atoms
library.list_atoms()  # ['AND', 'OR', 'XOR', ...]

# Verify (always passes - exact by construction)
passed, accuracy, failures = library.verify_atom("SUM")
```

### FP4Loader

```python
from trix.compiler.emit import FP4Loader

loader = FP4Loader()
circuit = loader.load(Path("./output/adder_8bit.trix.json"))

# Execute loaded circuit
outputs = circuit.execute(inputs)
```

---

## Available FP4 Atoms

| Atom | Inputs | Output | Implementation |
|------|--------|--------|----------------|
| AND | 2 | 1 | 1-layer threshold |
| OR | 2 | 1 | 1-layer threshold |
| NOT | 1 | 1 | 1-layer threshold |
| NAND | 2 | 1 | 1-layer threshold |
| NOR | 2 | 1 | 1-layer threshold |
| XOR | 2 | 1 | 2-layer minterm |
| XNOR | 2 | 1 | 2-layer minterm |
| SUM | 3 | 1 | 2-layer minterm |
| CARRY | 3 | 1 | 1-layer threshold |
| MUX | 3 | 1 | 2-layer minterm |

All atoms verified at **100% accuracy**.

---

## File Formats

### .trix.json

Standard TriX configuration (same as float32):

```json
{
  "name": "adder_8bit",
  "version": "1.0.0",
  "compiler_version": "0.1.0-fp4",
  "verified": true,
  "num_tiles": 16,
  "tile_configs": [...],
  "routes": [...],
  "execution_order": [...]
}
```

### .fp4 Weight Files

Custom 4-bit packed format:

```
Header:
  [4 bytes]  Magic: "TFP4"
  [1 byte]   Version: 1
  [1 byte]   Num layers
  [1 byte]   Name length
  [1 byte]   Num inputs
  [1 byte]   Num outputs
  [N bytes]  Name (UTF-8)

Per Layer:
  [2 bytes]  Weight rows (little-endian)
  [2 bytes]  Weight cols (little-endian)
  [2 bytes]  Bias count (little-endian)
  [N bytes]  Packed weights (2 values per byte)
  [M bytes]  Packed biases (2 values per byte)
```

Weight values are 4-bit indices into lookup tables:
- Weights: `[-1.0, 0.0, 1.0]`
- Biases: `[-2.5, -1.5, -0.5, 0.5, 1.5]`

---

## Packing Utilities

```python
from trix.compiler.fp4_pack import (
    pack_circuit,
    unpack_circuit,
    save_circuit,
    load_circuit,
    measure_sizes,
    verify_roundtrip,
)

# Pack a circuit
circuit = library.get_atom("SUM")
data = pack_circuit(circuit)

# Unpack
restored = unpack_circuit(data)

# Save/load files
save_circuit(circuit, Path("SUM.fp4"))
loaded = load_circuit(Path("SUM.fp4"))

# Measure compression
sizes = measure_sizes(circuit)
print(f"Float32: {sizes['float32_bytes']}B")
print(f"FP4: {sizes['fp4_bytes']}B")
print(f"Compression: {sizes['compression_ratio']:.1f}x")

# Verify round-trip
assert verify_roundtrip(circuit)
```

---

## Custom Atoms

Create custom FP4 atoms using the minterm generator:

```python
from trix.compiler import truth_table_to_circuit

# Define truth table
my_table = {
    (0, 0, 0): 0,
    (0, 0, 1): 1,
    (0, 1, 0): 1,
    (0, 1, 1): 0,
    (1, 0, 0): 1,
    (1, 0, 1): 0,
    (1, 1, 0): 0,
    (1, 1, 1): 1,
}

# Generate circuit
circuit = truth_table_to_circuit("MY_PARITY", 3, my_table)

# Verify
from trix.compiler.atoms_fp4 import verify_generated_circuit
passed, accuracy, failures = verify_generated_circuit(circuit, my_table)
assert passed  # 100% exact
```

---

## Comparison: Float32 vs FP4

| Aspect | Float32 | FP4 |
|--------|---------|-----|
| Accuracy | 100% (trained) | 100% (constructed) |
| Training | Required | Not needed |
| Convergence risk | Possible | None |
| Weight storage | 32 bits | 4 bits |
| Compression | 1x | ~7-8x (theoretical) |
| Actual compression | 1x | 1.7x (small circuits) |

Note: Actual compression is limited by header overhead for small circuits.

---

## Examples

### Full Adder

```python
compiler = TriXCompiler(use_fp4=True)
result = compiler.compile("full_adder")

# Test all 8 combinations
for a in [0, 1]:
    for b in [0, 1]:
        for cin in [0, 1]:
            out = result.execute({"A": a, "B": b, "Cin": cin})
            print(f"{a}+{b}+{cin} = {out['Sum']} (carry {out['Cout']})")
```

### 8-bit Adder

```python
compiler = TriXCompiler(use_fp4=True)
result = compiler.compile("adder_8bit")

def add(a, b):
    inputs = {"Cin": 0}
    for i in range(8):
        inputs[f"A[{i}]"] = (a >> i) & 1
        inputs[f"B[{i}]"] = (b >> i) & 1
    
    out = result.execute(inputs)
    sum_val = sum(out.get(f"Sum[{i}]", 0) << i for i in range(8))
    return sum_val, out.get("Cout", 0)

print(add(37, 28))   # (65, 0)
print(add(255, 1))   # (0, 1) - overflow!
```

### Save and Load

```python
# Compile and save
compiler = TriXCompiler(use_fp4=True)
result = compiler.compile("adder_8bit", output_dir="./my_adder")

# Later: load and use
from trix.compiler.emit import FP4Loader
loader = FP4Loader()
circuit = loader.load("./my_adder/adder_8bit.trix.json")
outputs = circuit.execute(inputs)
```

---

## Troubleshooting

### "Unknown atom: XXX"

FP4 only supports the 10 built-in atoms. Custom atoms must be created with `truth_table_to_circuit()`.

### Low compression ratio

Header overhead dominates for small circuits. Compression improves with larger circuits or more atom types.

### Verification shows 0% accuracy

Ensure you're using `use_fp4=True`. Float32 atoms need training; FP4 atoms are pre-verified.

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/trix/compiler/atoms_fp4.py` | FP4 atom library |
| `src/trix/compiler/fp4_pack.py` | Packing utilities |
| `src/trix/compiler/emit.py` | FP4Emitter, FP4Loader |
| `src/trix/compiler/compiler.py` | TriXCompiler with use_fp4 |
| `scripts/validate_fp4_atoms.py` | Validation script |
| `docs/FP4_ATOMS_RESULTS.md` | Detailed results |
| `docs/SPEC_FP4_ATOMS.md` | Original spec |

---

## Theory

### Why Threshold Circuits?

Boolean functions over {0,1} inputs can be implemented exactly using threshold logic:

1. **Linearly separable** functions (AND, OR, CARRY) need 1 layer
2. **Non-linearly separable** functions (XOR, SUM) use minterm detection

### Why Custom Encoding?

Standard FP4 formats (E2M1, NF4) don't include all our bias values (e.g., -2.5).

We use a custom 4-bit index encoding:
- 4-bit index → lookup table → exact float value
- Zero quantization error
- Simple decode (table lookup)

### The Key Insight

> "Don't learn the function. Construct it."

FP4 atoms aren't neural networks that learned boolean logic. They're boolean logic circuits implemented as neural networks.

The weights ARE the circuit. Training is construction.

---

*FP4: Exact computation in 4 bits.*
