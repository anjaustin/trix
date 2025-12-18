# Triton FFT Kernel - Convergence

*Pass 3: What wants to emerge? The architecture that assembled itself.*

---

## What Emerged

Across the passes, a clear architecture crystallized:

**TritonFFT: Compiled TriX for GPU**

The insight: TriX FFT is already correct. We're not changing the algorithm - we're changing the substrate from Python interpreter to GPU silicon.

```
TriX FFT (Python)     →    TritonFFT (GPU)
     │                          │
     ├─ ButterflyMicrocode      ├─ tl.* operations
     ├─ FFTRouter               ├─ Index calculations in kernel
     └─ Twiddle table           └─ Twiddle table (same)
```

Same algorithm. Same twiddles. Different execution engine.

---

## The Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TritonFFT                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                   │
│  │  Twiddle Table  │    │  Bit-Reverse    │                   │
│  │  (precomputed)  │    │  Indices        │                   │
│  └────────┬────────┘    └────────┬────────┘                   │
│           │                      │                             │
│           ▼                      ▼                             │
│  ┌─────────────────────────────────────────────────────┐      │
│  │              @triton.jit fft_kernel                  │      │
│  │                                                      │      │
│  │   for stage in tl.static_range(LOG_N):              │      │
│  │       stride = 1 << stage                           │      │
│  │       # Parallel butterflies                         │      │
│  │       W = twiddle_table[twiddle_idx]                │      │
│  │       a' = a + W * b                                │      │
│  │       b' = a - W * b                                │      │
│  │                                                      │      │
│  └─────────────────────────────────────────────────────┘      │
│                          │                                     │
│                          ▼                                     │
│                    [X_re, X_im]                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Decisions

### 1. Twiddle Strategy: Precomputed Table

**Decision:** Precompute twiddle factors, load to shared memory.

**Rationale:**
- Matches TriX "twiddle opcode" philosophy
- No runtime trig (np.cos, np.sin)
- Table size manageable: N complex values = 8N bytes
- N=1024 → 8KB (fits in shared memory)

### 2. Kernel Structure: Single Fused Kernel

**Decision:** One kernel does complete FFT (all stages).

**Rationale:**
- Avoid kernel launch overhead
- Use shared memory for intermediate stages
- `tl.static_range` for stage loop (compiles to unrolled code)

### 3. Parallelism: Per-Butterfly

**Decision:** Each thread block handles one FFT. Threads parallelize butterflies within stages.

**Rationale:**
- Natural mapping to Cooley-Tukey
- N/2 butterflies per stage → good thread utilization
- Batch dimension → multiple thread blocks

### 4. Precision: Float32 Default, Float64 Option

**Decision:** Float32 for speed, float64 for high-precision Riemann work.

**Rationale:**
- Float32 matches torch.fft to ~1e-6 (sufficient for most uses)
- Riemann Probe at very high t may need float64
- Triton supports both

---

## The Specification

### Class: TritonFFT

```python
class TritonFFT(nn.Module):
    """
    Triton-accelerated FFT with TriX philosophy.
    
    Features:
        - Precomputed twiddle table (no runtime trig)
        - Cooley-Tukey DIT algorithm
        - Batch support
        - Drop-in replacement for TriXFFT
    
    Performance target:
        - 10-50x over Python TriXFFT
        - Within 2-5x of torch.fft (acceptable)
    """
    
    def __init__(self, max_n: int = 1024):
        super().__init__()
        self.max_n = max_n
        self._twiddle_cache = {}
        self._bitrev_cache = {}
    
    def _get_twiddles(self, N: int, device) -> Tuple[Tensor, Tensor]:
        """Get precomputed twiddle factors for size N."""
        ...
    
    def _get_bitrev(self, N: int) -> Tensor:
        """Get bit-reversal permutation indices."""
        ...
    
    def forward(self, x_re: Tensor, x_im: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute FFT.
        
        Args:
            x_re: Real parts, shape (..., N) where N is power of 2
            x_im: Imaginary parts, shape (..., N)
        
        Returns:
            X_re, X_im: FFT output, same shape as input
        """
        ...
```

### Kernel: trix_fft_kernel

```python
@triton.jit
def trix_fft_kernel(
    # Input pointers
    X_re_ptr, X_im_ptr,
    # Output pointers  
    Y_re_ptr, Y_im_ptr,
    # Twiddle table pointers
    W_re_ptr, W_im_ptr,
    # Bit-reversal indices
    bitrev_ptr,
    # Dimensions
    N: tl.constexpr,
    LOG_N: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-kernel Cooley-Tukey DIT FFT.
    
    Each program handles one FFT in the batch.
    Uses shared memory for intermediate stages.
    """
    batch_idx = tl.program_id(0)
    
    # Load input with bit-reversal
    # ...
    
    # Cooley-Tukey stages
    for stage in tl.static_range(LOG_N):
        stride = 1 << stage
        group_size = stride << 1
        
        # Compute butterflies in parallel
        # ...
    
    # Store output
    # ...
```

---

## Implementation Plan

### Phase 1: N=8 Proof of Concept (1-2 hours)

1. Write minimal `trix_fft_kernel` for N=8
2. Hardcode LOG_N=3
3. Test against torch.fft
4. Benchmark vs Python TriXFFT

**Success criteria:** Correct results, measurable speedup.

### Phase 2: Generalization (2-3 hours)

1. Template kernel for LOG_N = 3, 4, 5, 6, 7, 8, 9, 10
2. Add batch dimension support
3. Create TritonFFT class wrapper
4. Full test suite

**Success criteria:** Works for N=8 to N=1024, batch sizes 1-1024.

### Phase 3: Integration (1 hour)

1. Create SpectralTileTriton using TritonFFT
2. Update RiemannProbeTriX to use new tile
3. Benchmark zeros/sec

**Success criteria:** ≥5x speedup over current TriX probe.

### Phase 4: Documentation

1. Update SPEC_RIEMANN_PROBE_TRIX.md
2. Add performance comparison to README
3. Document the TriX-to-Triton compilation pattern

---

## What This Proves

If successful, TritonFFT demonstrates:

1. **TriX compiles to GPU** - The pattern (structural routing + fixed microcode) works on GPU silicon, not just Python.

2. **Transparency preserved** - Triton source is readable. No black-box CUDA libraries.

3. **Performance recoverable** - We don't have to choose between TriX philosophy and speed.

4. **Twiddle opcodes generalize** - The "no runtime trig" principle applies at GPU level too.

---

## The Deeper Pattern

What emerged isn't just "a Triton FFT kernel." It's a template for TriX-to-GPU compilation:

```
TriX Module (Python)
     │
     ├─ Microcode (exact operations)     →  tl.* operations
     ├─ Router (structural selection)    →  Index calculations
     └─ Precomputed tables               →  Shared memory
     
     ▼
     
Triton Kernel (GPU)
```

This pattern could apply to:
- SpectralTile → TritonSpectral
- DirichletTile → TritonDirichlet  
- Any TriX tile with structural routing

TritonFFT is the prototype. If it works, we have a path to GPU-accelerated TriX across the board.

---

## Final Specification Summary

| Aspect | Decision |
|--------|----------|
| Algorithm | Cooley-Tukey DIT |
| Twiddles | Precomputed table, shared memory |
| Kernel structure | Single fused kernel |
| Parallelism | Batch × butterflies |
| Precision | Float32 default, float64 option |
| Target speedup | 10-50x over Python |
| API | Same as TriXFFT |

---

## Ready to Build

The specification is complete. The architecture is clear. The implementation path is defined.

Time to write code.

---

*End of convergence. Proceed to implementation.*
