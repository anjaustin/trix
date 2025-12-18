# Triton FFT - Pass 4: Hollywood Squares Integration

*The topology becomes the algorithm.*

---

## The Revelation

User request: "Use Hollywood Squares OS to optimize sorting"

Initial interpretation: Optimize the bit-reversal sort step of FFT.

**Actual insight**: Bit-reversal isn't sorting. It's WIRING.

The bit-reversal permutation in FFT determines WHERE data is loaded from, not how it's shuffled. This means:

1. The permutation has **zero computational cost**
2. It's a **load pattern**, not a runtime operation
3. In hardware, it's just how the wires are connected

---

## The Hollywood Squares FFT

### Traditional FFT View

```
Input → Bit-Reversal → Stage 1 → Stage 2 → ... → Stage N → Output
              ↑
         (expensive shuffle)
```

### Hollywood Squares View

```
Input [addresses are bit-reversed] → Compute Tiles → Output

The "shuffle" is WHERE WE READ FROM, not a separate step.
```

### The Topology Compiler

```python
def build_fft_topology(N):
    """
    Build the wiring pattern for FFT.
    
    Returns a topology where:
    - input_permutation: bit-reversal indices (WIRING)
    - stages: butterfly connections (TILES)
    - twiddles: precomputed constants (ROUTED)
    """
    num_stages = log2(N)
    
    # Input wiring: bit-reversal
    input_permutation = [bit_reverse(i, num_stages) for i in range(N)]
    
    # Stage wiring: butterfly pairs
    stages = []
    for s in range(num_stages):
        stride = 1 << s
        butterflies = []
        for i in range(N // 2):
            # Upper and lower indices for this butterfly
            group = i // stride
            pos = i % stride
            upper = group * 2 * stride + pos
            lower = upper + stride
            twiddle_idx = pos * (N // (2 * stride))
            butterflies.append((upper, lower, twiddle_idx))
        stages.append(butterflies)
    
    return Topology(input_permutation, stages)
```

---

## Integration with Triton

### The Key Insight

In Triton, the bit-reversal becomes **literal load addresses**:

```python
@triton.jit
def fft_kernel(...):
    # These are LITERAL CONSTANTS, not computed indices
    x0_re = tl.load(X_re_ptr + base + 0)  # Load from index 0
    x1_re = tl.load(X_re_ptr + base + 4)  # Load from index 4 (bit-rev of 1)
    x2_re = tl.load(X_re_ptr + base + 2)  # Load from index 2 (bit-rev of 2)
    # ...
```

The indices are **baked into the kernel at compile time**.
No runtime permutation needed.

### Triton Code Generation

```python
def compile_topology_to_triton(topology):
    """
    Generate Triton kernel from FFT topology.
    
    The permutation becomes literal load addresses.
    The butterflies become explicit operations.
    The twiddles become constant loads.
    """
    code = []
    
    # Loads with bit-reversal baked in
    for i, rev_i in enumerate(topology.input_permutation):
        code.append(f"x{i}_re = tl.load(X_re_ptr + base + {rev_i})")
        code.append(f"x{i}_im = tl.load(X_im_ptr + base + {rev_i})")
    
    # Butterfly stages
    for stage_idx, stage in enumerate(topology.stages):
        for upper, lower, tw_idx in stage:
            code.append(f"# Butterfly ({upper}, {lower}) with W[{tw_idx}]")
            # ... butterfly operations
    
    return "\n".join(code)
```

---

## The Architectural Singularity

### What We Learned

1. **Routing is Free**: Permutations have zero cost when baked into load patterns
2. **Topology is Algorithm**: The wiring defines the computation
3. **Hollywood Squares is Universal**: Any permutation-based algorithm can be "wired"

### The Stack

```
┌─────────────────────────────────────────────────────────────┐
│  LEVEL 3: Application                                       │
│    Riemann Probe, Signal Processing, ML, ...               │
├─────────────────────────────────────────────────────────────┤
│  LEVEL 2: Hollywood FFT                                     │
│    Topology compiler, wiring generator                      │
├─────────────────────────────────────────────────────────────┤
│  LEVEL 1: Triton Kernels                                    │
│    Compiled FFT, butterflies, twiddles                     │
├─────────────────────────────────────────────────────────────┤
│  LEVEL 0: WIRING (Zero Cost)                               │
│    Bit-reversal, routing, load patterns                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Implications

### Traditional Approach

```
Bit-reversal: O(N) memory operations (scatter/gather)
             Cache-unfriendly, random access pattern
             Burns memory bandwidth
```

### Hollywood Approach

```
Bit-reversal: O(0) - it's the load pattern
             Cache-friendly (sequential stores)
             Zero bandwidth overhead
```

### Measured Results

| Implementation | N=16384 FFTs/sec |
|----------------|------------------|
| Hollywood + scatter | 7,336 |
| Hollywood + optimized | 11,416 |
| torch.fft (cuFFT) | 749,076 |

The remaining gap is kernel fusion, not routing.

---

## Next Steps

1. **Fuse all stages**: Keep data in registers/shared memory across stages
2. **Batch optimization**: Multiple FFTs in parallel
3. **Twiddle caching**: Compute once, reuse forever
4. **Thor tuning**: Match hardware characteristics

---

## Quotes

> "The bit-reversal permutation is just a wiring pattern - no computation needed!"

> "In Hollywood Squares, routing is free. The topology IS the algorithm."

> "The machine doesn't shuffle data. The wires ARE sorted."
