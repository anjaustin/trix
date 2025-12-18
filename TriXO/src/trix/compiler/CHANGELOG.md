# TriX Compiler Changelog

All notable changes to the TriX Compiler will be documented in this file.

## [0.2.0] - 2025-12-16

### Added

FP4 Integration - Exact atoms in 4-bit packed format.

#### FP4 Atoms (`atoms_fp4.py`)
- 10 threshold circuit atoms: AND, OR, XOR, NOT, NAND, NOR, XNOR, SUM, CARRY, MUX
- All atoms verified at 100% accuracy
- Exact by construction (no training needed)
- Minterm generator for custom atoms

#### FP4 Packing (`fp4_pack.py`)
- Custom 4-bit encoding with lookup tables
- Pack/unpack utilities
- File I/O for `.fp4` weight files
- Zero quantization error

#### FP4 Emission (`emit.py`)
- `FP4Emitter` - Emits FP4-packed weight files
- `FP4Loader` - Loads FP4 circuits from disk
- `FP4CompiledCircuit` - Executable loaded circuit

#### Compiler Integration (`compiler.py`)
- `use_fp4=True` flag for FP4 mode
- Automatic backend selection
- Compatible with all existing templates

### Results

| Circuit | Status | Weight Size |
|---------|--------|-------------|
| Full Adder | 100% exact | 58 bytes |
| 8-bit Adder | 100% exact | 58 bytes |

---

## [0.1.0] - 2025-12-16

### Added

Initial release of the TriX Compiler.

#### Core Components

- **AtomLibrary** (`atoms.py`)
  - Pre-verified atomic operations: AND, OR, XOR, NOT, NAND, NOR, XNOR, SUM, CARRY, MUX
  - Atom training with exhaustive verification
  - Atom serialization (save/load)
  - Truth table-based verification

- **CircuitSpec** (`spec.py`)
  - Circuit specification language
  - Wire types: INPUT, OUTPUT, INTERNAL
  - Multi-bit wire support
  - Built-in templates: full_adder, adder_8bit, adder_16bit, adder_32bit, alu_1bit
  - JSON serialization

- **Decomposer** (`decompose.py`)
  - Circuit decomposition into atoms
  - Dependency graph analysis
  - Topological sort for execution order
  - Pattern decomposition for common circuits

- **Verifier** (`verify.py`)
  - Atom verification to 100% accuracy
  - Parallel verification support
  - Circuit-level verification
  - Exhaustive verification with oracle functions

- **Composer** (`compose.py`)
  - Tile allocation
  - Route generation (Hollywood Squares topology)
  - Signature generation for content-addressable routing
  - CircuitExecutor for runtime execution
  - Message tracing for debugging

- **Emitter** (`emit.py`)
  - TrixConfig generation
  - Weight file emission
  - Manifest with checksums
  - TrixLoader for loading compiled circuits
  - CompiledCircuit for execution

- **TriXCompiler** (`compiler.py`)
  - Main compiler orchestrating full pipeline
  - Template support
  - Verbose/quiet modes
  - compile_and_test helper

#### Demo

- `scripts/demo_compiler.py` - Full demonstration of compiler capabilities

### Technical Details

- Atoms are neural networks trained to 100% accuracy on bounded domains
- Circuits are composed using the Hollywood Squares topology model
- Execution follows topological order with deterministic message passing
- All verification is exhaustive (every input combination tested)

### Verified Results

| Circuit | Atoms | Verification |
|---------|-------|--------------|
| Full Adder | 2 | 100% (8 cases) |
| 8-bit Adder | 16 | 100% (all arithmetic) |
| Custom Circuits | Variable | 100% required |

---

## Architecture Notes

The compiler implements the "Neural Von Neumann" architecture:

```
Spec → Decompose → Verify → Compose → Emit
  │        │          │         │        │
Intent   Atoms    100% Exact  Topology  Files
```

Key insight: "The routing learns WHEN. The atoms compute WHAT."

This separates:
- **WHAT** (computation) - Verified atoms
- **WHEN** (dispatch) - Routing signatures  
- **HOW** (composition) - Topology wiring

Each layer has independent verification, enabling compositional correctness.
