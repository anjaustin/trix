# Triton FFT Kernel - Exploration & Reflection

*Pass 2: Nodes of opportunity. Engineering lens. What survives constraint?*

---

## Threads That Keep Appearing

From Pass 1, these themes recurred:

1. **"We have all the pieces"** - twiddle opcodes, Triton infra, FFT algorithm
2. **"Butterfly parallelism"** - N/2 independent ops per stage
3. **"Structural routing"** - index calculations are deterministic
4. **"TriX compiled to GPU"** - same philosophy, different substrate
5. **"Is the juice worth the squeeze?"** - effort vs payoff question

---

## Node 1: The Minimal Viable Kernel

**What if we just wrote the simplest possible Triton FFT?**

Constraints:
- Fixed N (start with N=8 or N=16)
- Unrolled stages (constexpr LOG_N)
- Precomputed twiddle table
- Single kernel for complete FFT

Engineering reality check:
- Triton supports complex numbers? No, need real/imag pairs
- Stage fusion in single kernel? Yes, with shared memory
- Performance expectation? Probably 10-100x over Python, but maybe 0.5-2x vs torch.fft

**Verdict: BUILDABLE. Start here.**

---

## Node 2: Stage-by-Stage Kernel

**What if each stage is a separate kernel call?**

Constraints:
- One kernel = one stage
- Memory traffic: read array, write array, repeat
- Log2(N) kernel launches

Engineering reality check:
- Kernel launch overhead ~5-20μs each
- For N=1024, 10 launches = 50-200μs overhead
- For large batch sizes, overhead amortizes

**Verdict: TOO SLOW for small FFTs. Maybe OK for large batches.**

---

## Node 3: Batched FFT

**What if we batch many FFTs together?**

The Riemann Probe evaluates Z(t) at MANY t values. If we're doing 100K evaluations, we could:
- Batch them into groups of 1024
- Each group is one FFT
- Parallelize across groups

This is how torch.fft handles batches - the batch dimension parallelizes.

Engineering reality check:
- Triton handles 2D parallelism well (program_id(0), program_id(1))
- Each program handles one FFT in the batch
- Within-FFT parallelism for butterflies

**Verdict: THIS IS THE PATH. Batch over t values.**

---

## Node 4: Use Existing Triton FFT

**What if someone already wrote a good Triton FFT?**

Quick search of my memory:
- triton-lang examples have basic FFT
- FasterTransformer has optimized kernels
- Flash-FFT project exists

But:
- External dependencies
- May not match our twiddle opcode philosophy
- Harder to verify/understand

**Verdict: CHECK FIRST, but probably write our own for transparency.**

---

## Node 5: The Odlyzko-Schönhage Integration

**What if we don't need general FFT at all?**

The Riemann Probe uses FFT for a specific purpose: evaluating Z(t) at uniformly spaced points.

The algorithm is:
1. Compute coefficients a_n for n=1..N
2. FFT the coefficient array
3. Read off Z(t_k) values from FFT output

Maybe we can write a specialized kernel that does exactly this, not general FFT.

Engineering reality check:
- Specialization often wins over generalization
- The coefficient computation is already TriX (DirichletTile)
- Just need the "apply FFT" part

**Verdict: INTERESTING. Consider after general FFT works.**

---

## Node 6: Compile TriX FFT to Triton

**What if we auto-generate Triton code from TriX FFT?**

The TriX FFT has:
- ButterflyMicrocode (exact ops)
- FFTRouter (deterministic indices)
- TriXFFT (composition)

Could we "compile" this to Triton?
- Router → index calculations in kernel
- Microcode → tl operations
- Composition → kernel structure

Engineering reality check:
- This is essentially writing a TriX-to-Triton compiler
- High effort, high abstraction
- Might be the "right" long-term solution

**Verdict: TOO AMBITIOUS for now. Defer.**

---

## Priority Stack

1. **Minimal N=8 Triton FFT** - prove the concept works
2. **Scale to N=64, N=256** - useful sizes
3. **Batch support** - the real performance win
4. **Integrate with SpectralTile** - actual usage
5. **Specialize for Riemann** - if needed

---

## Engineering Lens: What Actually Matters?

### Correctness
- Must match torch.fft to 1e-6 (float32) or 1e-14 (float64)
- Twiddle table approach ensures this
- Verification is straightforward

### Performance
- Target: at least 5x over Python Cooley-Tukey
- Stretch: within 2x of torch.fft
- Measure: zeros/sec in Riemann Probe

### Transparency
- Readable Triton source
- No magic constants
- Clear correspondence to Cooley-Tukey algorithm

### Integration
- Drop-in replacement for TriXFFT.forward()
- Same API: (x_re, x_im) → (X_re, X_im)
- Device-aware (cuda tensors)

---

## The Spec Taking Shape

```python
class TritonFFT:
    """
    Triton-accelerated FFT with TriX philosophy.
    
    - Precomputed twiddle table (no runtime trig)
    - Cooley-Tukey DIT algorithm
    - Compiled to GPU via Triton
    """
    
    def __init__(self, max_n: int = 1024):
        self.max_n = max_n
        self._twiddle_cache = {}
    
    def forward(self, x_re: torch.Tensor, x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute FFT.
        
        Args:
            x_re: Real parts, shape (..., N)
            x_im: Imaginary parts, shape (..., N)
        
        Returns:
            X_re, X_im: FFT output
        """
        # Get or compute twiddles
        # Call Triton kernel
        # Return results
```

---

## What Survives Constraint?

After the engineering lens:

1. ✅ **Single-kernel FFT** - buildable, verifiable
2. ✅ **Precomputed twiddles** - matches TriX philosophy
3. ✅ **Batch support** - essential for Riemann performance
4. ❌ **Stage-by-stage kernels** - too much overhead
5. ⏸️ **Auto-compilation** - defer to future
6. ⏸️ **Specialization for Riemann** - defer until needed

---

## The Implementation Path

### Phase 1: Proof of Concept
- N=8 hardcoded Triton FFT
- Compare to torch.fft
- Measure performance

### Phase 2: Generalization
- Template for powers of 2
- N=16, 32, 64, 128, 256, 512, 1024
- Batch dimension support

### Phase 3: Integration
- TritonFFT class with same API as TriXFFT
- Drop into SpectralTile
- Benchmark Riemann Probe

### Phase 4: Optimization (if needed)
- Shared memory usage
- Warp-level primitives
- Register blocking

---

## Questions Remaining

1. What's the actual performance of existing Triton FFT implementations?
2. How does batch size affect performance?
3. Is float32 sufficient for Riemann Probe at high t?
4. Should we support in-place FFT?

---

## Gut Check

Does this feel right?

The approach is:
- Conservative (start small, verify, scale)
- Pragmatic (accept we won't beat cuFFT)
- TriX-aligned (transparent, verifiable, no runtime trig)

Yes, this feels right. The goal isn't to beat cuFFT - it's to have a TriX-native FFT that's fast enough for practical use while maintaining our principles.

---

*End of exploration. Ready for Pass 3: Convergence.*
