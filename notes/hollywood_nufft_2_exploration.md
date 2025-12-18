# Hollywood Squares × NUFFT: Exploration

## Testing the Inverse Mapping Idea

```python
# Current: scatter (source → destinations)
for n in range(M):
    for offset in [-1, 0, 1, 2]:
        m = floor(grid_idx[n]) + offset
        grid[m] += a_n[n] * weight(grid_idx[n] - m)

# Hollywood: gather (destination ← sources)  
for m in range(num_grid):  # PARALLEL!
    for n in sources_for[m]:
        grid[m] += a_n[n] * weight(grid_idx[n] - m)
```

## Building the Inverse Mapping

For grid point m, which n values contribute?

n contributes to m if: m - 2 < grid_idx[n] < m + 2

Since grid_idx[n] = delta * ln(n) / delta_omega:
- n_min: grid_idx[n] = m - 2  →  n = exp((m-2) * delta_omega / delta)
- n_max: grid_idx[n] = m + 2  →  n = exp((m+2) * delta_omega / delta)

The contributing n values form a CONTIGUOUS RANGE!

## This is Huge

If sources form contiguous ranges, we don't need sparse indexing.
Just store: (n_start[m], n_end[m]) for each grid point m.

Memory: O(2 × num_grid) = O(num_grid)
Not O(M × 4)!

## Let Me Verify

delta_omega = 2π / num_grid
grid_idx[n] = delta * ln(n) / delta_omega = delta * ln(n) * num_grid / (2π)

For typical values:
- t0 = 10^9, M = 12625
- delta = 0.1 / density ≈ 0.1 / 2.3 ≈ 0.043
- num_grid = 8 * M = 101000
- delta_omega = 2π / 101000 ≈ 6.2e-5

grid_idx[1] = 0.043 * 0 / 6.2e-5 = 0
grid_idx[M] = 0.043 * ln(12625) / 6.2e-5 = 0.043 * 9.44 / 6.2e-5 = 6550

So grid indices range from 0 to ~6550 (out of 101000 grid points).
Most grid points have NO sources!

## Sparse Structure

Only grid points 0 to ~6550 have any sources.
Grid points 6551 to 100999 are all zero.

This is because omega_max = delta * ln(M) < 2π.

The FFT of mostly-zeros is wasteful!

## Wait, This Changes Everything

We're doing FFT of size 101000, but only 6550 non-zero inputs.
That's 6.5% density.

For the gather approach:
- Only need to compute 6550 grid points
- FFT: still need full size for correct phase relationships

Hmm, but the output is also sparse in some sense...

## Back to the Gather Kernel

Let's implement and benchmark:

```python
def build_gather_map(M, delta, delta_omega, num_grid):
    """Build the inverse mapping: grid_idx → source range."""
    # grid_idx[n] = delta * ln(n) / delta_omega
    # n contributes to m if m-2 < grid_idx[n] < m+2
    # Equivalently: n in [exp((m-2)*delta_omega/delta), exp((m+2)*delta_omega/delta)]
    
    scale = delta_omega / delta
    
    n_lo = torch.zeros(num_grid, dtype=torch.long)
    n_hi = torch.zeros(num_grid, dtype=torch.long)
    
    for m in range(num_grid):
        # n range that contributes to grid point m
        lo = max(1, int(math.exp((m - 2) * scale)))
        hi = min(M, int(math.exp((m + 2) * scale)) + 1)
        n_lo[m] = lo
        n_hi[m] = hi
    
    return n_lo, n_hi


def gather_spread(a_n, grid_idx, n_lo, n_hi, num_grid):
    """Gather-based spreading."""
    grid = torch.zeros(num_grid, dtype=a_n.dtype, device=a_n.device)
    
    for m in range(num_grid):
        if n_hi[m] > n_lo[m]:
            ns = torch.arange(n_lo[m], n_hi[m])
            dists = grid_idx[ns-1] - m  # grid_idx is 0-indexed
            weights = cubic_bspline(dists)
            grid[m] = (a_n[ns-1] * weights).sum()
    
    return grid
```

## The Contiguous Range Advantage

Since sources are contiguous, we can use:
- Vectorized loads (coalesced memory access)
- Reduction operations (parallel sum)
- No index indirection

```python
# Triton kernel sketch
@triton.jit
def gather_spread_kernel(grid_ptr, a_n_ptr, grid_idx_ptr, n_lo_ptr, n_hi_ptr, ...):
    m = tl.program_id(0)
    
    lo = tl.load(n_lo_ptr + m)
    hi = tl.load(n_hi_ptr + m)
    
    # Load contiguous range of sources
    ns = lo + tl.arange(0, MAX_SOURCES)
    mask = ns < hi
    
    a_vals = tl.load(a_n_ptr + ns - 1, mask=mask, other=0)
    gidx = tl.load(grid_idx_ptr + ns - 1, mask=mask, other=0)
    
    # Compute weights
    dist = gidx - m
    w = cubic_bspline(dist)
    
    # Sum
    result = tl.sum(a_vals * w, axis=0)
    tl.store(grid_ptr + m, result)
```

## Problem: Variable Range Sizes

Different grid points have different range sizes.
- Grid point near 0: many sources (n=1,2,3,... all have small ln)
- Grid point near max: few sources

This creates load imbalance in the kernel.

## Solution: Binning by Range Size

Group grid points by their range size:
- Bin 1: range size 1-4
- Bin 2: range size 5-8
- Bin 3: range size 9-16
- ...

Launch separate kernels for each bin with appropriate MAX_SOURCES.

Or: use a single kernel with MAX_SOURCES = max over all grid points.
Wasteful for small ranges, but simpler.

## What's the Actual Range Distribution?

n contributes to m if: m - 2 < grid_idx[n] < m + 2
Range width in n: exp((m+2)*scale) - exp((m-2)*scale) 
                = exp(m*scale) * (exp(2*scale) - exp(-2*scale))
                = exp(m*scale) * 2*sinh(2*scale)

For small scale, sinh(2*scale) ≈ 2*scale, so range ≈ 4*scale*exp(m*scale)

Range grows EXPONENTIALLY with m!
- Small m (low freq): small range
- Large m (high freq): large range

Wait, this is backwards from my intuition. Let me reconsider.

## Reconsider: n → grid_idx mapping

grid_idx[n] = delta * ln(n) / delta_omega

Low n → low grid_idx
High n → high grid_idx

For grid point m, sources are n where grid_idx[n] ≈ m.
That means n ≈ exp(m * delta_omega / delta).

Range of n: [exp((m-2)*scale), exp((m+2)*scale)] where scale = delta_omega/delta

At m=0: n in [exp(-2*scale), exp(2*scale)] - centered at n=1
At m=100: n in [exp(98*scale), exp(102*scale)] - much wider in absolute terms

So higher grid indices (higher freq) have MORE sources.
This makes sense: ln(n) changes slowly for large n, so many n map to similar omega.

## Implications for Load Balancing

Grid points with high index have more sources.
But there are also fewer grid points with high index (above omega_max there are none).

The work distribution is:
- Many grid points with 0 sources (above omega_max)
- Some grid points with few sources (low m)
- Some grid points with many sources (high m, approaching omega_max)

## Sparse Approach

Only process grid points that have sources!

active_grid_points = {m : 0 <= m <= grid_idx[M] + 2}

This reduces work from O(num_grid) to O(omega_max / delta_omega) = O(M * oversampling).

## Final Insight

The Hollywood approach + sparsity awareness could give us:
1. No atomics (gather instead of scatter)
2. Better memory access (contiguous source ranges)
3. Less work (only process active grid points)

Expected speedup: 2-4x from eliminating atomics and improving memory patterns.

## To Implement

1. Compute active grid range: [0, ceil(grid_idx[M]) + 2]
2. For each active grid point, compute source range
3. Gather kernel with contiguous loads
4. FFT on full grid (pad with zeros)

Let's code it.
