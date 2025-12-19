# TriX Compiler

**Compile high-level specifications into verified neural circuits.**

The TriX Compiler transforms circuit specifications into executable neural networks that compute **exactly** - not approximately, not probabilistically, but with 100% accuracy on their bounded domains.

## Philosophy

> "The routing learns WHEN. The atoms compute WHAT."

Traditional neural networks are trained end-to-end to approximate functions. The TriX Compiler takes a different approach:

1. **Decompose** complex functions into atomic operations
2. **Verify** each atom achieves 100% accuracy
3. **Compose** verified atoms into circuits
4. **Inherit** correctness from the composition

This is not machine learning. This is **neural compilation**.

## Quick Start

```python
from trix.compiler import TriXCompiler

# Create compiler (float32 atoms, trained)
compiler = TriXCompiler()

# Or use FP4 atoms (exact by construction, 4-bit packed)
compiler = TriXCompiler(use_fp4=True)

# Compile a circuit
result = compiler.compile("adder_8bit")

# Check verification
print(f"Verified: {result.verification.all_verified}")

# Execute
inputs = {"A[0]": 1, "B[0]": 1, "Cin": 0, ...}
outputs = result.execute(inputs)

# Save to disk
result = compiler.compile("adder_8bit", output_dir="./output")
```

## Pipeline

```
┌─────────┐    ┌───────────┐    ┌────────┐    ┌─────────┐    ┌──────┐
│  Spec   │ -> │ Decompose │ -> │ Verify │ -> │ Compose │ -> │ Emit │
└─────────┘    └───────────┘    └────────┘    └─────────┘    └──────┘
     │              │               │              │             │
 CircuitSpec   Atom Types      100% Exact     Topology      Files
```

### Stage 1: Specification

Define your circuit using `CircuitSpec` or use a built-in template:

```python
# Use template
result = compiler.compile("full_adder")

# Or define custom
spec = CircuitSpec("my_circuit")
spec.add_input("A")
spec.add_input("B")
spec.add_output("Y")
spec.add_atom("and_gate", "AND", ["A", "B"], ["Y"])
result = compiler.compile(spec)
```

### Stage 2: Decomposition

The decomposer analyzes the circuit and determines:
- Which atom types are needed
- The dependency graph between atoms
- Execution order (topological sort)

### Stage 3: Verification

Each atom type is trained and verified:
- Exhaustive test cases (all input combinations)
- Training until 100% accuracy
- Verification certificate

**If any atom fails to reach 100%, compilation fails.**

### Stage 4: Composition

Verified atoms are wired together:
- Tile allocation (one tile per atom instance)
- Route generation (Hollywood Squares topology)
- Signature generation (content-addressable routing)

### Stage 5: Emission

The compiled circuit is saved:
- `.trix.json` - Topology and routing configuration
- `weights/*.pt` - Trained neural atom weights
- `manifest.json` - Checksums for verification

## Available Atoms

| Atom | Inputs | Output | Operation |
|------|--------|--------|-----------|
| AND | 2 | 1 | Logical AND |
| OR | 2 | 1 | Logical OR |
| XOR | 2 | 1 | Logical XOR |
| NOT | 1 | 1 | Logical NOT |
| NAND | 2 | 1 | Logical NAND |
| NOR | 2 | 1 | Logical NOR |
| XNOR | 2 | 1 | Logical XNOR |
| SUM | 3 | 1 | Full adder sum (A⊕B⊕Cin) |
| CARRY | 3 | 1 | Full adder carry |
| MUX | 3 | 1 | 2:1 Multiplexer |

## Circuit Templates

| Template | Description | Atoms |
|----------|-------------|-------|
| `full_adder` | 1-bit full adder | SUM, CARRY |
| `adder_8bit` | 8-bit ripple carry adder | 16× (SUM, CARRY) |
| `adder_16bit` | 16-bit ripple carry adder | 32× (SUM, CARRY) |
| `adder_32bit` | 32-bit ripple carry adder | 64× (SUM, CARRY) |
| `alu_1bit` | 1-bit ALU (ADD/AND/OR/XOR) | Multiple |

## API Reference

### TriXCompiler

```python
compiler = TriXCompiler(
    library=None,      # AtomLibrary (created if None)
    cache_dir=None,    # Cache for trained atoms
    verbose=True       # Print progress
)

result = compiler.compile(
    spec_or_name,      # CircuitSpec or template name
    output_dir=None,   # Emit files here (optional)
    verify_exhaustive=False,
    oracle=None        # Function for exhaustive verification
)
```

### CompilationResult

```python
result.success          # bool - all atoms verified?
result.spec             # CircuitSpec
result.decomposition    # DecompositionResult
result.verification     # CircuitVerificationReport
result.topology         # Topology
result.execute(inputs)  # Execute the circuit
result.summary()        # Human-readable summary
```

### CircuitSpec

```python
spec = CircuitSpec("name", "description")
spec.add_input("A", width=1)
spec.add_output("Y", width=1)
spec.add_internal("temp", width=1)
spec.add_atom("instance_name", "ATOM_TYPE", ["in1", "in2"], ["out1"])
spec.validate()         # Check for errors
spec.save(path)         # Save to JSON
spec.load(path)         # Load from JSON
```

## Architecture

```
trix/compiler/
├── __init__.py     # Public API
├── atoms.py        # Atom, AtomLibrary, AtomNetwork
├── spec.py         # CircuitSpec, Wire, AtomInstance
├── decompose.py    # Decomposer, PatternDecomposer
├── verify.py       # Verifier, ExhaustiveVerifier
├── compose.py      # Composer, Topology, CircuitExecutor
├── emit.py         # Emitter, TrixConfig, TrixLoader
└── compiler.py     # TriXCompiler (main entry point)
```

## Theory

The TriX Compiler is based on several key insights:

### 1. Atomic Decomposition

Complex functions can be decomposed into simple atoms. Each atom is simple enough that a neural network can learn it **exactly**.

### 2. Compositional Correctness

If each atom is 100% correct, and the wiring is correct, the whole circuit is correct. This is the Hollywood Squares theorem:

> Deterministic message passing + bounded local semantics + enforced observability ⇒ global convergence with inherited correctness

### 3. Neural Compilation

Training isn't teaching the network to approximate. It's **configuring** the network to **be** the function. The weights are the circuit. The inference is the computation.

## FP4 Mode

FP4 mode uses threshold circuits instead of trained neural networks.

### Why FP4?

| Aspect | Float32 | FP4 |
|--------|---------|-----|
| Training | Required | Not needed |
| Convergence | Risk of failure | Guaranteed |
| Storage | 32 bits/weight | 4 bits/weight |
| Accuracy | 100% (if trained) | 100% (by construction) |

### How It Works

FP4 atoms are **constructed**, not trained:

```python
# AND gate as threshold circuit
AND(a,b) = step(a + b - 1.5)

# Weights: {1, 1}, Bias: {-1.5}
# All values fit in 4-bit encoding
```

### Usage

```python
compiler = TriXCompiler(use_fp4=True)
result = compiler.compile("adder_8bit", output_dir="./output")

# Emitted files:
#   ./output/adder_8bit.trix.json  (topology)
#   ./output/weights/SUM.fp4       (35 bytes)
#   ./output/weights/CARRY.fp4     (23 bytes)
```

### Custom FP4 Atoms

```python
from trix.compiler import truth_table_to_circuit

# Any boolean function can become an FP4 atom
my_table = {(0,0): 0, (0,1): 1, (1,0): 1, (1,1): 0}
circuit = truth_table_to_circuit("MY_XOR", 2, my_table)
```

See `docs/FP4_INTEGRATION.md` for complete documentation.

---

## Related Projects

- **TriX** - The neural architecture (tiles, routing, signatures)
- **FLYNNCONCEIVABLE** - Proof that neural networks can be exact CPUs
- **Hollywood Squares OS** - Coordination layer for verified composition

## License

MIT
