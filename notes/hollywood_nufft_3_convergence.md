# Hollywood Squares × NUFFT: Convergence

## The Core Insight

**Scatter → Gather transformation eliminates atomics**

Current (scatter): Each source n writes to 4 grid points → COLLISIONS → atomics
Hollywood (gather): Each grid point reads from its sources → NO COLLISIONS → pure parallel

## Key Discovery: Contiguous Source Ranges

For grid point m, contributing sources n form a CONTIGUOUS range:
```
n ∈ [exp((m-2) × scale), exp((m+2) × scale)]
where scale = delta_omega / delta
```

This means:
- No sparse indexing needed
- Just store (n_lo, n_hi) per grid point
- Coalesced memory loads

## Implementation Plan

### 1. Pre-compute Gather Map (One-time, O(num_grid))

```python
def build_gather_map(M, delta, delta_omega, num_grid):
    scale = delta_omega / delta
    max_grid = min(num_grid, int(delta * math.log(M) / delta_omega) + 3)
    
    n_lo = torch.zeros(max_grid, dtype=torch.int32)
    n_hi = torch.zeros(max_grid, dtype=torch.int32)
    
    for m in range(max_grid):
        lo = max(1, int(math.exp((m - 2) * scale)))
        hi = min(M + 1, int(math.exp((m + 2) * scale)) + 1)
        n_lo[m] = lo
        n_hi[m] = hi
    
    return n_lo, n_hi, max_grid
```

### 2. Gather Kernel (Triton)

```python
@triton.jit
def gather_spread_kernel(
    grid_re_ptr, grid_im_ptr,
    a_re_ptr, a_im_ptr,
    grid_idx_ptr,
    n_lo_ptr, n_hi_ptr,
    max_grid: tl.constexpr,
    MAX_RANGE: tl.constexpr,  # Max sources per grid point
):
    m = tl.program_id(0)
    if m >= max_grid:
        return
    
    lo = tl.load(n_lo_ptr + m)
    hi = tl.load(n_hi_ptr + m)
    
    # Load contiguous source range
    offsets = tl.arange(0, MAX_RANGE)
    ns = lo + offsets
    mask = ns < hi
    
    # Load source values (0-indexed, so ns-1)
    a_re = tl.load(a_re_ptr + ns - 1, mask=mask, other=0.0)
    a_im = tl.load(a_im_ptr + ns - 1, mask=mask, other=0.0)
    gidx = tl.load(grid_idx_ptr + ns - 1, mask=mask, other=0.0)
    
    # B-spline weights
    dist = gidx - m
    t = dist - tl.floor(dist)
    # ... cubic B-spline computation ...
    
    # Parallel reduction
    sum_re = tl.sum(a_re * w, axis=0)
    sum_im = tl.sum(a_im * w, axis=0)
    
    tl.store(grid_re_ptr + m, sum_re)
    tl.store(grid_im_ptr + m, sum_im)
```

### 3. Sparse FFT Handling

Only `max_grid` points are non-zero (typically 6-10% of full grid).
Options:
- Pad to full grid, do standard FFT (current approach, simple)
- Use sparse FFT library (complex, marginal gain)

Recommendation: Keep standard FFT for now. The gather optimization is the main win.

## Actual Results

After profiling with Triton-fused kernels:

| Step | Time | % of Total |
|------|------|------------|
| Coefficients | 0.24ms | 5.6% |
| Grid indices | 0.52ms | 12.1% |
| B-spline weights | 0.71ms | 16.6% |
| **Scatter** | **0.73ms** | **17.0%** |
| **FFT** | **1.55ms** | **36.3%** |
| Theta+Z | 0.53ms | 12.4% |

**Key finding: FFT is the bottleneck, not scatter!**

After Triton fusion: **1.83ms total** → FFT is now **85%** of execution time.

## Why Gather Didn't Help

1. Scatter is already fast (0.73ms, only 17%)
2. Vectorized gather creates large intermediate tensors (slow)
3. The Triton scatter kernel is already well-optimized
4. FFT dominates anyway

## Real Insight

The Hollywood "routing is free" applies to FFT itself - the butterfly operations
ARE the routing. cuFFT already optimizes this perfectly.

To go faster, we need to either:
1. Reduce FFT size (hurts accuracy)
2. Avoid FFT entirely (different algorithm)
3. Sparse FFT (only compute needed outputs)

## Testing Plan

1. Implement gather-based spreading in Python (verify correctness)
2. Implement Triton kernel
3. Benchmark against scatter-based approach
4. Verify zero detection accuracy maintained

## What We Tried

### 1. Gather vs Scatter
- Gather (destination reads from sources): **SLOWER** (12.9ms vs 0.73ms)
- Creates large intermediate tensors
- Not a win

### 2. Pre-computed Topology (Hollywood Indices)
- Pre-compute scatter indices and B-spline weights once
- **4.77x speedup on spreading** (0.468ms → 0.098ms)
- **1.21x overall speedup** (4.06M → 4.91M zeros/s)
- **THIS WORKS!**

### 3. Sparse Matrix Multiply
- Build CSR sparse spreading matrix
- cusparse mm: **SLOWER** than scatter
- CSR overhead kills performance

### 4. Batched Evaluation
- Multiple t0 values through same topology
- Scatter loop serializes batches - **NO SPEEDUP**
- Sparse mm batching: still slower than scatter

## Final Results

| Approach | Spreading Time | Overall Rate |
|----------|----------------|--------------|
| Original | 0.468ms | 4.06M zeros/s |
| **Hollywood Pre-computed** | **0.098ms** | **4.91M zeros/s** |
| Sparse mm | 3.05ms | ~2M zeros/s |
| Batched scatter | ~6ms/batch | ~4.7M zeros/s |

## Key Takeaways

1. **Pre-computing topology is the right approach**
   - Indices + weights fixed for nearby t0 values
   - Amortize computation over many evaluations

2. **FFT is the true bottleneck** (85% of fused time)
   - Spreading optimization has diminishing returns
   - Need to either reduce FFT size or use sparse FFT

3. **Batching doesn't help with current scatter**
   - Scatter_add has no batch dimension
   - Loop over batches serializes work

4. **Sparse matrix formats have overhead**
   - Our sparsity pattern (4 entries/column) is not ideal for CSR
   - Direct scatter is better optimized

## Recommendation

Use the **Hollywood Pre-computed Engine** with:
- One-time topology build (indices + weights)
- Fast scatter_add with pre-computed data
- Standard cuFFT (already optimal)

Rate: **~5M zeros/sec** at t=10^10

## Code

See `hollywood_gather_nufft.py` for implementation.
