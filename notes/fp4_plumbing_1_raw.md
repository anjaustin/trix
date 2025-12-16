# FP4 Plumbing: Raw Thoughts

*Stream of consciousness on what needs to happen to wire FP4 atoms into the compiler.*

---

## What We Have

Two atom libraries:
1. `AtomLibrary` - trains neural networks to 100% (float32)
2. `FP4AtomLibrary` - constructs threshold circuits (FP4-compatible values in float32)

The compiler currently uses AtomLibrary. It trains atoms, verifies them, composes them, emits them.

FP4AtomLibrary has verified atoms but isn't wired in.

---

## The Gap

The values are FP4-compatible, but they're stored as float32. There's no actual 4-bit packing.

So what does "FP4 plumbing" actually mean?

1. **Compiler integration** - `use_fp4=True` flag that swaps AtomLibrary for FP4AtomLibrary
2. **Weight representation** - Store weights as actual 4-bit packed values
3. **Emission format** - Save packed weights to disk
4. **Loading** - Load packed weights back
5. **Inference** - Execute with packed weights (unpack on the fly or use FP4 kernels)

---

## The Easy Parts

**Compiler integration** is trivial. The interfaces are similar. Just swap the library.

**Emission** is mostly format work. JSON config stays the same. Weight files change format.

---

## The Hard Parts

**Actual 4-bit packing.** How do we represent this?

Options:
- Pack 2 FP4 values per byte (straightforward)
- Use existing libraries (bitsandbytes has 4-bit support)
- Custom format

For our atoms, weights are {-1, 0, +1} and biases are {-2.5, -1.5, -0.5, 0.5, 1.5}. That's only ~8 distinct values total. Could use 3-bit encoding!

But let's stick with standard FP4 for compatibility.

**Inference path.** Two options:
1. Unpack to float32 at load time, run normal inference
2. Stay packed, use quantized kernels

Option 1 is simpler. We still get storage compression. Just not compute savings.

Option 2 requires custom kernels. More complex. Bigger payoff.

For now: Option 1. Get it working. Optimize later.

---

## The Interfaces

Current flow:
```
TriXCompiler
  -> AtomLibrary.get_atom(name) -> AtomNetwork (nn.Module)
  -> Verifier.verify_atom() 
  -> Composer.compose() -> Topology with atom references
  -> Emitter.emit() -> .trix.json + weights/*.pt
```

FP4 flow:
```
TriXCompiler(use_fp4=True)
  -> FP4AtomLibrary.get_atom(name) -> ThresholdCircuit
  -> Verifier needs to work with ThresholdCircuit
  -> Composer needs to work with ThresholdCircuit
  -> Emitter needs to emit packed FP4 weights
```

The mismatch: ThresholdCircuit isn't an nn.Module. It's a dataclass.

We have FP4AtomModule that wraps it. But do we need that?

Actually, for the compiler, we just need:
- Something that can execute (forward pass)
- Something we can get weights from

ThresholdCircuit has both. We can make it work.

---

## The Weight Packing Question

Current atom weights (SUM, 2-layer):
```
Layer 0: W=[[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [1, 1, 1]], b=[-0.5, -0.5, -0.5, -2.5]
Layer 1: W=[[1, 1, 1, 1]], b=[-0.5]
```

In FP4 (E2M1), we can represent:
- -1, 0, 1 ✓
- -0.5, 0.5, 1.5, -1.5, -2.5... wait, -2.5?

Let me check E2M1 range again:
```
E2M1: sign(1) + exp(2) + mantissa(1) = 4 bits
Values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
```

-2.5 is NOT in E2M1! It's between 2 and 3.

Hmm. But it is in NF4:
```
NF4 values (from QLoRA):
{-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0}
```

Wait, NF4 maxes out at ±1.0. -2.5 isn't there either!

So our bias values don't actually fit in standard FP4 formats.

**This is a problem.**

---

## Rethinking

The biases are the issue. Weights are {-1, 0, +1} which fit fine.

Biases are {-2.5, -1.5, -0.5, 0.5, 1.5}.

Options:
1. Store biases as FP8 or FP16 (mixed precision)
2. Scale the whole circuit so biases fit in FP4
3. Use a custom 4-bit encoding for our specific value set

Option 3 is cleanest for our use case. We have exactly 5 bias values. That's 3 bits. We have 8 weight values if you count all used. Still fits in 4 bits with a lookup table.

**Custom encoding:**
```
Index | Weight Value
------+-------------
0     | -1.0
1     | 0.0
2     | 1.0
3     | (unused)

Index | Bias Value
------+-------------
0     | -2.5
1     | -1.5
2     | -0.5
3     | 0.5
4     | 1.5
5-7   | (unused)
```

Actually, let's just use a lookup table. Store 4-bit indices, dequantize via table lookup.

This is simpler than trying to fit into E2M1/NF4.

---

## Revised Plan

1. **Custom FP4 encoding** for atom weights/biases
   - 4-bit indices into a lookup table
   - Table has our exact values
   - No quantization error

2. **Packing format**
   - 2 values per byte
   - Separate tables for weights vs biases

3. **Emission**
   - Pack weights as bytes
   - Include lookup tables in config

4. **Loading**
   - Read packed bytes
   - Dequantize via table lookup
   - Execute as float32

This gets us storage compression without precision loss.

---

## What I Need to Build

1. `fp4_pack.py` - Packing/unpacking utilities
2. Update `Emitter` to use FP4 packing for FP4 atoms
3. Update `TrixLoader` to unpack FP4 weights
4. Add `use_fp4` flag to `TriXCompiler`
5. Test end-to-end

---

## Questions Remaining

- Should Verifier work with ThresholdCircuit directly, or wrap in Module first?
- How to handle the Composer? It references atoms by type, gets them from library.
- Is the current Topology structure compatible with FP4 atoms?

Need to trace through the code to see exactly what changes.

---

*End raw thoughts. Time to explore systematically.*
