# Triton FFT Kernel - Raw Thoughts

*Pass 1: Stream of consciousness. No filter. Surface area over coherence.*

---

## The Problem Space

We have a 100% TriX Riemann Probe. It works. It's correct. It's slow.

5K zeros/sec vs 20K+ with PyTorch. That's a 4x gap minimum. The README claims 475K zeros/sec for zeta_fft.py which uses torch.fft. So potentially 100x gap.

The bottleneck is obvious: pure Python Cooley-Tukey FFT. Loop over stages, loop over butterflies, Python interpreter overhead at every step.

But wait - we already have the pieces:
- Twiddle opcodes (no runtime trig)
- FP4 atoms (threshold circuits)
- Triton dequant kernel (working)
- FFT compiler that compiles routing to FP4

Why aren't we using them together?

---

## What We Have (Mental Inventory)

1. **TriX FFT** (spectral_tile.py)
   - Cooley-Tukey DIT
   - Algorithmic routing (not learned)
   - ButterflyMicrocode (exact)
   - Works perfectly, just slow

2. **Twiddle Opcodes** (fft_compiler.py)
   - TWIDDLE_TABLE_8 - algebraic constants
   - make_twiddle_op() - creates fixed complex multiply
   - get_twiddle_index() - structural routing
   - No np.cos, no np.sin at runtime

3. **Triton Infrastructure** (benchmark_triton_dequant.py)
   - @triton.jit decorator works
   - tl.program_id, tl.arange, tl.load, tl.store
   - We know how to write Triton kernels

4. **FP4 Atoms** (trix_cuda.py)
   - SUMAtom, CARRYAtom
   - Threshold circuits
   - Mesa 8 proved they work

---

## The Gap

We have TriX FFT (slow, correct) and torch.fft (fast, opaque).

The middle ground - Triton FFT with TriX philosophy - doesn't exist yet.

What would it look like?

---

## Stream of Consciousness

Triton compiles to PTX which runs on CUDA cores. It's not cuFFT but it's fast. People have written Triton FFTs before - there's precedent.

The question is: can we write a Triton FFT that:
1. Uses our twiddle opcode philosophy (no runtime trig)
2. Maintains TriX transparency (readable, verifiable)
3. Gets competitive performance with torch.fft

Let me think about the structure...

A Triton FFT kernel would:
- Take real and imaginary arrays
- Do bit-reversal permutation
- Loop through stages (log2(N) of them)
- At each stage, compute butterflies in parallel
- Output the transformed arrays

The parallelism is at the butterfly level. Within a stage, all butterflies are independent.

For N=1024:
- Stage 0: 512 butterflies
- Stage 1: 512 butterflies
- ...
- Stage 9: 512 butterflies

Each stage has N/2 butterflies. That's 512 parallel operations at each stage.

On a GPU with 10K+ threads, we can do all butterflies in a stage simultaneously.

---

## The Twiddle Question

cuFFT precomputes twiddle tables in shared memory. We could do the same:
- Precompute TWIDDLE_TABLE[k] = (cos(-2πk/N), sin(-2πk/N))
- Load into shared memory at kernel start
- Index into table during butterfly

This is exactly our "twiddle opcode" philosophy - the selection is structural, the values are fixed.

But wait - for large N, the table gets big. N=1024 means 1024 complex values = 8KB.
For N=4096, that's 32KB. Still fits in shared memory.

Alternative: compute twiddles on the fly but cache in registers for each thread block.

Actually, the smarter approach: use the recursive property. W_N^k = W_N^(k mod N). 
And W_{2N}^{2k} = W_N^k. So we can build larger twiddle sets from smaller ones.

But this is premature optimization. Start simple.

---

## Simplest Triton FFT

```python
@triton.jit
def trix_fft_kernel(
    X_re_ptr, X_im_ptr,  # Input
    Y_re_ptr, Y_im_ptr,  # Output
    W_re_ptr, W_im_ptr,  # Twiddle table
    N: tl.constexpr,
    LOG_N: tl.constexpr,
):
    # Bit-reverse permutation
    # ...
    
    # Cooley-Tukey stages
    for stage in range(LOG_N):
        stride = 1 << stage
        # Parallel butterflies
        # ...
```

The challenge: Triton doesn't like variable loop bounds well. LOG_N needs to be constexpr.

This means we'd need separate kernels for each N, or use tl.static_range.

Actually, looking at existing Triton FFT implementations... they often unroll the stages.

---

## What About the Riemann Probe Specifically?

The Riemann Probe doesn't actually use FFT in the critical path right now!

Looking at spectral_tile.py:
```python
# Step 3: Sum the series (direct summation - FFT variant TODO)
sum_real = real.sum(dim=1)
sum_imag = imag.sum(dim=1)
```

It's doing direct summation, not FFT-accelerated evaluation!

So the current bottleneck isn't even FFT - it's the serial evaluation of Z(t).

The FFT acceleration would be for evaluating Z(t) at many points simultaneously using the Odlyzko-Schönhage algorithm.

Wait, let me reconsider...

---

## Where FFT Actually Helps Riemann

The Odlyzko-Schönhage algorithm uses FFT to evaluate:
- Z(t_0), Z(t_0 + δ), Z(t_0 + 2δ), ..., Z(t_0 + (N-1)δ)

Instead of computing each Z(t) separately (N × O(√t) work), you can use FFT to get all N values in O(N log N + √t) time.

This is a huge win for scanning the critical line.

The zeta_fft.py implementation does this with torch.fft.

So the question becomes: can we replace torch.fft with Triton FFT in that algorithm?

---

## Going in Circles

I notice I'm circling between:
1. "Write a general Triton FFT kernel"
2. "Optimize specifically for Riemann Probe"
3. "Use existing twiddle opcode infrastructure"

Let me try to find the common thread...

The common thread is: **FFT is embarrassingly parallel at the butterfly level**.

Whether we're doing general FFT or Riemann-specific, the core operation is the same butterfly:
```
a' = a + W*b
b' = a - W*b
```

And the control (which W, which a, which b) is structural.

---

## What Wants to Emerge?

I think what wants to emerge is:

**A Triton butterfly kernel that:**
1. Takes arrays + twiddle table
2. Does one stage of FFT
3. Runs butterflies in parallel

Then we compose stages by calling the kernel log2(N) times.

This keeps things simple and modular. Each kernel call is one stage.

The bit-reversal can be a separate kernel or done in the first stage.

---

## Concerns

1. **Kernel launch overhead**: log2(N) kernel launches per FFT. For N=1024, that's 10 launches. Might be slow.
   - Mitigation: Fuse multiple stages into one kernel
   - Mitigation: Use persistent kernel pattern

2. **Memory bandwidth**: Reading/writing full arrays at each stage.
   - Mitigation: Use shared memory for intermediate stages
   - Mitigation: Fuse stages that fit in shared memory

3. **Twiddle access patterns**: Different threads need different twiddles.
   - Mitigation: Preload twiddle table to shared memory
   - Mitigation: Compute twiddles on the fly (but we wanted to avoid this!)

4. **Numerical precision**: We're using float32 currently.
   - The TriX FFT matches torch.fft to 1e-6, which is fine
   - For Riemann, we might want float64 at high t values

---

## The TriX Philosophy Angle

TriX says: "Routing learns WHEN. Atoms compute WHAT."

For FFT:
- WHEN = which elements pair, which twiddle to use (structural, deterministic)
- WHAT = butterfly operation (exact arithmetic)

A Triton FFT kernel would have:
- Structural routing compiled into the kernel (the index calculations)
- Exact arithmetic in the butterfly (a + W*b, a - W*b)

This IS TriX philosophy, just compiled to GPU code instead of Python.

The transparency comes from:
- Readable Triton source (not CUDA C)
- No learned parameters
- Deterministic behavior
- Verifiable against torch.fft

---

## Next Steps (If I Were to Build This)

1. **Start minimal**: N=8 Triton FFT, single kernel, hardcoded stages
2. **Verify correctness**: Compare to torch.fft and numpy.fft
3. **Benchmark**: Is it actually faster than Python Cooley-Tukey?
4. **Scale up**: N=64, N=256, N=1024
5. **Integrate**: Use in SpectralTile for Riemann Probe
6. **Optimize**: Fused stages, shared memory, etc.

---

## What I'm Uncertain About

- Is Triton the right tool? Maybe we should use raw CUDA or cupy.
- Will kernel launch overhead kill us?
- How do existing Triton FFT implementations compare to cuFFT?
- Is the Riemann Probe even FFT-bound, or is it something else?

---

## Raw Ending

I've touched a lot of surface area. The core insight is:
- We have all the pieces (twiddle opcodes, Triton infrastructure, FFT algorithm)
- We just haven't wired them together
- A Triton FFT would be "compiled TriX" - same philosophy, GPU speed

The question is whether the juice is worth the squeeze. If Triton FFT is only 2x faster than Python but torch.fft is 10x faster, maybe we just accept the torch.fft dependency for production and keep TriX FFT for verification/transparency.

But that feels like giving up. The whole point of TriX is that we don't need opaque libraries.

Let me sit with this...

---

*End of raw thoughts. Ready for Pass 2.*
