# FP4 Plumbing: Exploration

*Systematically working through the details identified in raw thoughts.*

---

## Key Discovery: Bias Values Don't Fit Standard FP4

This was the critical finding. Our biases {-2.5, -1.5, -0.5, 0.5, 1.5} don't fit in E2M1 or NF4.

**Resolution:** Custom lookup table encoding.

This is actually *better* than standard FP4 because:
- Zero quantization error (exact values)
- Simpler decode (table lookup vs FP4 math)
- Still 4 bits per value

---

## The Value Sets

### Weights
```
Used: {-1.0, 0.0, 1.0}
Indices: 0=-1.0, 1=0.0, 2=1.0
Bits needed: 2
```

### Biases  
```
Used: {-2.5, -1.5, -0.5, 0.5, 1.5}
Indices: 0=-2.5, 1=-1.5, 2=-0.5, 3=0.5, 4=1.5
Bits needed: 3
```

### Combined Encoding
```
4-bit value (0-15):
  0-2:   Weights (-1, 0, +1)
  3-7:   Biases (-2.5, -1.5, -0.5, 0.5, 1.5)
  8-15:  Reserved/unused
```

Or separate tables for weights and biases (cleaner).

---

## Packing Format

### Option A: Nibble Packing
Two 4-bit values per byte:
```
byte = (high_nibble << 4) | low_nibble
```

Pack weights sequentially, then biases.

### Option B: Bit-Level Packing
Weights need 2 bits, biases need 3 bits. Could pack more tightly.

But nibble packing is simpler and standard. Go with Option A.

---

## File Format

Current `.trix.json` + `weights/*.pt` (PyTorch format)

New for FP4:
```
circuit.trix.json      # Same topology, metadata
weights/
  SUM.fp4              # Packed 4-bit weights
  CARRY.fp4            # Packed 4-bit weights
  ...
manifest.json          # Checksums, includes value tables
```

### FP4 Weight File Format
```
Header (16 bytes):
  magic: "TFP4" (4 bytes)
  version: uint8
  num_layers: uint8
  reserved: 10 bytes

Per Layer:
  weight_rows: uint16
  weight_cols: uint16
  bias_size: uint16
  weight_data: ceil(rows*cols/2) bytes (nibble packed)
  bias_data: ceil(bias_size/2) bytes (nibble packed)

Footer:
  weight_table: float32[3] = {-1.0, 0.0, 1.0}
  bias_table: float32[5] = {-2.5, -1.5, -0.5, 0.5, 1.5}
```

Actually, the tables are fixed. Just hardcode them in the loader.

---

## Code Changes Needed

### 1. New File: `fp4_pack.py`

```python
WEIGHT_TABLE = [-1.0, 0.0, 1.0]
BIAS_TABLE = [-2.5, -1.5, -0.5, 0.5, 1.5]

def weight_to_index(w: float) -> int:
    """Map weight value to 4-bit index."""
    
def bias_to_index(b: float) -> int:
    """Map bias value to 4-bit index."""

def pack_nibbles(values: List[int]) -> bytes:
    """Pack 4-bit values into bytes."""

def unpack_nibbles(data: bytes, count: int) -> List[int]:
    """Unpack bytes into 4-bit values."""

def pack_layer(weights: Tensor, bias: Tensor) -> bytes:
    """Pack a layer's weights and biases."""

def unpack_layer(data: bytes, shape: Tuple) -> Tuple[Tensor, Tensor]:
    """Unpack a layer from bytes."""

def pack_circuit(circuit: ThresholdCircuit) -> bytes:
    """Pack entire circuit."""

def unpack_circuit(data: bytes, name: str) -> ThresholdCircuit:
    """Unpack circuit from bytes."""
```

### 2. Update: `emit.py`

Add method to Emitter:
```python
def emit_fp4_weights(self, circuit: ThresholdCircuit, path: Path):
    """Emit FP4-packed weights."""
```

Modify `emit()` to detect FP4 atoms and use FP4 emission.

### 3. Update: `compiler.py`

Add `use_fp4` parameter:
```python
class TriXCompiler:
    def __init__(self, use_fp4: bool = False, ...):
        self.use_fp4 = use_fp4
        if use_fp4:
            self.library = FP4AtomLibrary()
        else:
            self.library = AtomLibrary()
```

### 4. Update: `compose.py`

The Composer creates CircuitExecutor which holds atoms. Need to handle both AtomNetwork and ThresholdCircuit.

Currently:
```python
self._atoms[atom_type] = library.get_atom(atom_type)
```

This should work if FP4AtomLibrary.get_atom() returns something callable. ThresholdCircuit is callable. Good.

### 5. New/Update: Loading

Add FP4 loading to TrixLoader:
```python
def load_fp4_weights(self, path: Path) -> ThresholdCircuit:
    """Load FP4-packed weights."""
```

---

## Interface Compatibility Check

### AtomLibrary Interface
```python
library.get_atom(name) -> AtomNetwork (nn.Module, callable)
library.train_atom(name) -> trains and returns
library.list_atoms() -> List[str]
```

### FP4AtomLibrary Interface
```python
library.get_atom(name) -> ThresholdCircuit (callable)
library.list_atoms() -> List[str]
library.verify_atom(name) -> (passed, accuracy, failures)
```

Differences:
- No `train_atom` (not needed - atoms are constructed)
- `verify_atom` returns different type

For compiler integration, we need:
- `get_atom(name)` - ✓ both have it
- `list_atoms()` - ✓ both have it
- Callable return from get_atom - ✓ both work

The Verifier expects certain interface. Let me check...

---

## Verifier Compatibility

Current Verifier does:
```python
atom = library.get_atom(atom_type)
# ... uses atom as nn.Module for forward pass
```

ThresholdCircuit isn't nn.Module but is callable. The verification just needs forward pass.

May need small adapter or update Verifier to handle both.

---

## Execution Path Trace

```
TriXCompiler.compile(spec)
  1. Decomposer analyzes spec -> atom types needed
  2. Verifier verifies each atom type
     - Gets atom from library
     - Runs exhaustive test
     - For FP4: atoms already verified by construction
  3. Composer creates topology
     - Creates CircuitExecutor with atoms
     - Atoms are called during execute()
  4. Emitter saves to disk
     - For FP4: use FP4 packing
  5. Return CompilationResult
     - Has execute() method using CircuitExecutor
```

Most of this should work with FP4 atoms. Key changes:
- Verifier: handle ThresholdCircuit
- Emitter: FP4 packing
- Loader: FP4 unpacking

---

## Simplification Opportunity

The FP4 atoms are **already verified by construction**. We don't need to re-verify them.

Could add flag to skip verification for FP4:
```python
if self.use_fp4:
    # Skip verification - atoms are exact by construction
    verification = CircuitVerificationReport(all_verified=True, ...)
else:
    verification = self.verifier.verify_circuit(...)
```

But keeping verification is good for sanity checking. Just make sure it works.

---

## Memory Savings Calculation

Current (float32):
- SUM atom: 4*3 + 4 + 1*4 + 1 = 12 + 4 + 4 + 1 = 21 floats = 84 bytes
- CARRY atom: 1*3 + 1 = 4 floats = 16 bytes

FP4 packed:
- SUM atom: ceil(12/2) + ceil(4/2) + ceil(4/2) + ceil(1/2) = 6 + 2 + 2 + 1 = 11 bytes
- CARRY atom: ceil(3/2) + ceil(1/2) = 2 + 1 = 3 bytes

Compression for SUM: 84/11 = 7.6x
Compression for CARRY: 16/3 = 5.3x

For an 8-bit adder (8 SUM + 8 CARRY):
- Float32: 8*(84+16) = 800 bytes
- FP4: 8*(11+3) = 112 bytes
- Compression: 7.1x

Close to the theoretical 8x!

---

## Implementation Order

1. `fp4_pack.py` - Core packing utilities
2. Test packing round-trip
3. Update Emitter for FP4
4. Update TrixLoader for FP4
5. Add `use_fp4` to TriXCompiler
6. End-to-end test
7. Measure actual compression

---

*Exploration complete. Ready to implement.*
