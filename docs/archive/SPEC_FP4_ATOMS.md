# Spec: FP4 Atom Implementation

**Goal:** Implement TriX atoms using FP4 weights while maintaining 100% accuracy.

**Status:** COMPLETE ✓  
**Priority:** High  
**Predecessor:** TriX Compiler (v0.7.0)

> **UPDATE (2025-12-16):** Implementation complete. See `docs/FP4_ATOMS_RESULTS.md` for results.
> All 10 atoms verified at 100% accuracy using threshold circuit construction.

---

## Context

### What We Built

The TriX Compiler transforms circuit specifications into verified neural circuits:

```
Spec → Decompose → Verify → Compose → Emit
```

Key result: **Neural circuits that compute exactly** (not approximately).

Example: An 8-bit adder composed from SUM and CARRY atoms achieves 100% accuracy on all arithmetic operations.

### The Gap

Current atoms use **float32 weights**. They achieve 100% accuracy, but:
- Float32 is memory-inefficient
- The original TriX vision is 2-bit/4-bit weights
- FP4 would give 8x memory compression over float32

### The Opportunity

Atoms are **tiny**:
- 2-3 inputs
- 1 output  
- 4-8 truth table entries

If FP4 can nail these small functions exactly, the compiler handles the rest through composition.

---

## Background: What Are Atoms?

Atoms are the smallest verified computational units. Each atom implements a simple boolean or arithmetic function.

### Current Atom Library

| Atom | Inputs | Output | Function | Truth Table Size |
|------|--------|--------|----------|------------------|
| AND | 2 | 1 | a ∧ b | 4 |
| OR | 2 | 1 | a ∨ b | 4 |
| XOR | 2 | 1 | a ⊕ b | 4 |
| NOT | 1 | 1 | ¬a | 2 |
| NAND | 2 | 1 | ¬(a ∧ b) | 4 |
| NOR | 2 | 1 | ¬(a ∨ b) | 4 |
| XNOR | 2 | 1 | ¬(a ⊕ b) | 4 |
| SUM | 3 | 1 | a ⊕ b ⊕ c | 8 |
| CARRY | 3 | 1 | maj(a,b,c) | 8 |
| MUX | 3 | 1 | s ? b : a | 8 |

### Current Implementation

```python
class AtomNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim=32):
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs),
            nn.Sigmoid()
        )
```

This achieves 100% with float32. The question: **Can FP4 do the same?**

---

## The Task

### Objective

Create FP4-compatible atom implementations that:
1. Use FP4 weights (4-bit floating point)
2. Achieve **100% accuracy** on exhaustive truth tables
3. Integrate with the existing compiler infrastructure

### Success Criteria

```
For each atom in {AND, OR, XOR, NOT, NAND, NOR, XNOR, SUM, CARRY, MUX}:
  accuracy = test_all_input_combinations(atom)
  assert accuracy == 1.0, f"{atom} failed: {accuracy}"
```

**100% is not a target. It's a requirement.** 99.9% is failure.

### Deliverables

1. `src/trix/compiler/atoms_fp4.py` - FP4 atom implementations
2. `scripts/validate_fp4_atoms.py` - Verification script
3. `docs/FP4_ATOMS_RESULTS.md` - Results documentation
4. Integration with `TriXCompiler` (optional flag to use FP4 atoms)

---

## Technical Approach

### Option 1: Quantization-Aware Training (QAT)

Train with FP4 constraints from the start:

```python
class FP4Linear(nn.Module):
    def forward(self, x):
        # Quantize weights to FP4 during forward pass
        w_fp4 = quantize_to_fp4(self.weight)
        return F.linear(x, w_fp4, self.bias)
    
    def backward(self, grad):
        # Straight-through estimator for gradients
        ...
```

**Pros:** Network learns within FP4 constraints  
**Cons:** May not converge to 100%

### Option 2: Post-Training Quantization with Verification

1. Train float32 atom to 100%
2. Quantize to FP4
3. Verify accuracy
4. If < 100%, try different quantization parameters
5. If still failing, flag atom as FP4-incompatible

```python
def quantize_and_verify(atom, truth_table):
    for scale in search_space:
        fp4_atom = quantize(atom, scale=scale)
        acc = verify(fp4_atom, truth_table)
        if acc == 1.0:
            return fp4_atom
    return None  # Failed
```

**Pros:** Simple, uses existing trained atoms  
**Cons:** May not find valid quantization

### Option 3: Increased Hidden Dimension

FP4 has less precision. Compensate with more parameters:

```python
# Float32: hidden_dim=32 works
# FP4: try hidden_dim=64, 128, 256
```

**Pros:** More capacity may absorb quantization error  
**Cons:** Reduces efficiency gains

### Option 4: Architecture Search

Try different network architectures:
- Deeper networks (more layers)
- Different activations (ReLU, GELU, hardtanh)
- Residual connections
- Different initializations

```python
architectures = [
    {"layers": 2, "hidden": 32, "activation": "relu"},
    {"layers": 3, "hidden": 64, "activation": "gelu"},
    {"layers": 2, "hidden": 128, "activation": "hardtanh"},
    ...
]

for arch in architectures:
    atom = build_atom(arch)
    train_fp4(atom)
    if verify(atom) == 1.0:
        return atom
```

### Option 5: Special Encodings

Instead of raw binary inputs, use encodings:

**Thermometer encoding:**
```
0 → [0, 0, 0, 0]
1 → [1, 1, 1, 1]
```

**One-hot encoding:**
```
(0,0) → [1, 0, 0, 0]
(0,1) → [0, 1, 0, 0]
(1,0) → [0, 0, 1, 0]
(1,1) → [0, 0, 0, 1]
```

**Pros:** May make function easier to learn in low precision  
**Cons:** Increases input dimension

---

## FP4 Format Reference

### E2M1 Format (common FP4)

```
[sign][exp][exp][mantissa]
  1     2    2      1 bit
```

**Representable values:** 
```
±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
```

That's only 16 distinct values (including ±0).

### NF4 Format (from QLoRA)

Normalized float4, optimized for neural network weight distributions:
```
{-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0}
```

### Implementation Note

You may need to implement custom FP4 quantization. Libraries:
- `bitsandbytes` (has NF4)
- `quanto` (Hugging Face)
- Custom implementation

---

## Verification Infrastructure

### Existing Tools

The compiler already has verification:

```python
from trix.compiler import AtomLibrary
from trix.compiler.verify import Verifier

library = AtomLibrary()
verifier = Verifier(library)

# Verify single atom
result = verifier.verify_atom("SUM")
print(f"Accuracy: {result.accuracy}")  # Must be 1.0

# Verify all atoms
report = verifier.verify_all()
print(report.summary())
```

### Adding FP4 Verification

```python
# Your new code
from trix.compiler.atoms_fp4 import FP4AtomLibrary

fp4_library = FP4AtomLibrary()
verifier = Verifier(fp4_library)

for atom_name in fp4_library.list_atoms():
    result = verifier.verify_atom(atom_name)
    status = "PASS" if result.accuracy == 1.0 else "FAIL"
    print(f"{atom_name}: {status} ({result.accuracy:.2%})")
```

---

## Recommended Order of Attack

### Phase 1: Baseline (Day 1)

1. Read `src/trix/compiler/atoms.py` - understand current implementation
2. Run `scripts/demo_compiler.py` - see the system work
3. Implement naive FP4 quantization of existing atoms
4. Measure accuracy drop

**Expected outcome:** Most atoms will drop below 100%

### Phase 2: QAT Exploration (Days 2-3)

1. Implement FP4 quantization-aware training
2. Train each atom from scratch with FP4 constraints
3. Track which atoms reach 100%, which don't

**Expected outcome:** Some atoms may work, others may need more effort

### Phase 3: Architecture Search (Days 4-5)

For atoms that failed:
1. Try increased hidden dimension
2. Try different architectures
3. Try special encodings

**Expected outcome:** Find configurations that work for each atom

### Phase 4: Integration (Day 6)

1. Create `FP4AtomLibrary` with all working atoms
2. Add `use_fp4=True` flag to `TriXCompiler`
3. Verify full pipeline with FP4 atoms
4. Document results

---

## Code Pointers

### Key Files

```
src/trix/compiler/
├── atoms.py        # Current atom implementation (READ THIS FIRST)
├── verify.py       # Verification engine
├── compiler.py     # Main compiler
└── __init__.py     # Public API

scripts/
├── demo_compiler.py  # Full demo (RUN THIS FIRST)
```

### Entry Points

```python
# Train an atom
from trix.compiler.atoms import AtomLibrary
library = AtomLibrary()
library.train_atom("SUM")  # Trains and verifies

# Get atom network
atom = library.get_atom("SUM")  # Returns nn.Module

# Verify
from trix.compiler.verify import Verifier
v = Verifier(library)
result = v.verify_atom("SUM")
```

### Truth Tables

```python
# Truth tables are defined in atoms.py
TRUTH_TABLES = {
    "AND": {
        (0, 0): (0,),
        (0, 1): (0,),
        (1, 0): (0,),
        (1, 1): (1,),
    },
    "SUM": {
        (0, 0, 0): (0,),
        (0, 0, 1): (1,),
        (0, 1, 0): (1,),
        (0, 1, 1): (0,),
        (1, 0, 0): (1,),
        (1, 0, 1): (0,),
        (1, 1, 0): (0,),
        (1, 1, 1): (1,),
    },
    # ...
}
```

---

## Success Metrics

### Must Have

- [ ] All 10 atoms achieve 100% accuracy with FP4 weights
- [ ] Verification passes exhaustively (not sampled)
- [ ] Integration with existing compiler

### Nice to Have

- [ ] Memory usage reduction measured (should be ~8x vs float32)
- [ ] Inference speed comparison
- [ ] Documentation of which approaches worked/failed

### Stretch Goals

- [ ] FP4 atoms work for 16-bit and 32-bit adder compositions
- [ ] Custom FP4 kernel for inference
- [ ] Atom training time < 1 second per atom

---

## Questions?

### Resources

- `notes/SESSION_INDEX.md` - Overview of the compiler session
- `src/trix/compiler/README.md` - Compiler documentation
- `CHANGELOG.md` - Project history (see v0.7.0)

### Key Insight

> "The routing learns WHEN. The atoms compute WHAT."

You're working on the WHAT. Make the atoms exact in FP4, and the compiler handles composition.

### The Bar

**100% accuracy is not negotiable.** 

The entire architecture depends on atom exactness. If an atom is 99% accurate, errors compound through composition. A 16-bit adder with 32 atoms at 99% each would have only 72% accuracy overall.

Exactness is the foundation. Don't compromise it.

---

*Good luck. The compiler is waiting for FP4 atoms.*
