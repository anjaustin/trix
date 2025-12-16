# FP4 Plumbing: Convergence

*Synthesis into implementation plan.*

---

## The Solution

**Custom lookup table encoding** for atom weights and biases.

Not E2M1. Not NF4. A simple 4-bit index into fixed value tables.

```
Weight Table: [-1.0, 0.0, 1.0]
Bias Table:   [-2.5, -1.5, -0.5, 0.5, 1.5]
```

### Why This Works

1. **Exact values** - No quantization error
2. **Simple decode** - Table lookup, not FP4 math
3. **4-bit storage** - Same compression as FP4
4. **Fast** - Integer index → float lookup

---

## Implementation Plan

### Step 1: Create `fp4_pack.py`

Core utilities for packing/unpacking:

```python
# Value tables
WEIGHT_TABLE = torch.tensor([-1.0, 0.0, 1.0])
BIAS_TABLE = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5])

# Packing
def pack_weights(tensor: Tensor) -> bytes
def pack_biases(tensor: Tensor) -> bytes
def pack_circuit(circuit: ThresholdCircuit) -> bytes

# Unpacking  
def unpack_weights(data: bytes, shape: tuple) -> Tensor
def unpack_biases(data: bytes, size: int) -> Tensor
def unpack_circuit(data: bytes, name: str) -> ThresholdCircuit
```

### Step 2: Update Emitter

Add FP4 emission path:

```python
def emit_fp4(self, topology: Topology, output_dir: Path):
    # Emit .trix.json (same as before)
    # Emit weights/*.fp4 (packed format)
```

### Step 3: Update Loader

Add FP4 loading:

```python
def load_fp4(self, path: Path) -> CompiledCircuit:
    # Load .trix.json
    # Load and unpack weights/*.fp4
    # Return executable circuit
```

### Step 4: Wire into Compiler

```python
class TriXCompiler:
    def __init__(self, use_fp4: bool = False, ...):
        self.use_fp4 = use_fp4
        self.library = FP4AtomLibrary() if use_fp4 else AtomLibrary()
```

### Step 5: End-to-End Test

```python
compiler = TriXCompiler(use_fp4=True)
result = compiler.compile("adder_8bit", output_dir="./fp4_output")

# Verify output
loaded = TrixLoader.load("./fp4_output")
assert loaded.execute(inputs) == expected

# Measure compression
float_size = measure_float_weights()
fp4_size = measure_fp4_weights()
print(f"Compression: {float_size / fp4_size:.1f}x")
```

---

## File Format

### `.fp4` Weight File

```
[4 bytes]  Magic: "TFP4"
[1 byte]   Version: 1
[1 byte]   Num layers
[2 bytes]  Reserved

Per layer:
  [2 bytes]  Weight rows
  [2 bytes]  Weight cols  
  [2 bytes]  Bias count
  [N bytes]  Packed weights (nibble pairs)
  [M bytes]  Packed biases (nibble pairs)
```

Simple. Self-describing. Easy to parse.

---

## Compression Estimate

| Circuit | Float32 | FP4 | Ratio |
|---------|---------|-----|-------|
| Full Adder | 100 B | 14 B | 7.1x |
| 8-bit Adder | 800 B | 112 B | 7.1x |
| 16-bit Adder | 1.6 KB | 224 B | 7.1x |
| 32-bit Adder | 3.2 KB | 448 B | 7.1x |

Consistent ~7x compression. Close to theoretical 8x (limited by header overhead on small circuits).

---

## What We're NOT Doing (Yet)

1. **FP4 compute kernels** - We unpack to float32 for execution
2. **On-device packing** - Pack offline, unpack at load
3. **Mixed precision** - All values use same encoding

These are optimizations for later. First: get it working.

---

## Success Criteria

1. [ ] `compiler.compile("adder_8bit", use_fp4=True)` works
2. [ ] Emits `.fp4` weight files
3. [ ] Loads and executes correctly
4. [ ] All test cases pass (100% exact)
5. [ ] Measured compression ~7x

---

## Execute

Time to write the code.

Order:
1. `fp4_pack.py` (new file)
2. Test packing round-trip
3. Update `emit.py`
4. Update loader in `emit.py`
5. Update `compiler.py`
6. End-to-end test

---

*Plan complete. Executing.*
