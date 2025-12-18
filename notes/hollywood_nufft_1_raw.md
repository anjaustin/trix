# Hollywood Squares × NUFFT: Raw Exploration

## The Problem We're Stuck On

scatter_add is killing us. 20% of time. Atomic operations. Random memory access.

```python
for i, offset in enumerate([-1, 0, 1, 2]):
    idx = (idx_floor + offset) % num_grid
    grid.scatter_add_(0, idx, a_n * weights[i])
```

Each source n writes to 4 grid points. GPU threads collide. Atomics serialize.

## Hollywood Squares Insight

"Routing is free."

In FFT, bit-reversal looks like sorting. But in hardware, it's just WIRING. 
Connect output 5 to input 3. Done. Zero cycles.

The permutation IS the memory layout.

## Flipping the Script

Current thinking: source → destinations (scatter)
- n=1 writes to grid[0,1,2,3]
- n=2 writes to grid[1,2,3,4]  
- n=3 writes to grid[1,2,3,4]  (collision with n=2!)
- Collisions require atomics

Hollywood thinking: destination ← sources (gather)
- grid[0] reads from n=1
- grid[1] reads from n=1,2,3
- grid[2] reads from n=1,2,3,4
- No collisions! Each output is independent!

## Wait... Can We Invert?

B-spline spreading: each n contributes to 4 grid points
B-spline gathering: each grid point receives from ~4 sources

But which sources? The mapping is:
- grid_idx[n] = delta * ln(n) / delta_omega
- n contributes to floor(grid_idx[n]) + {-1, 0, 1, 2}

Inverting: which n values have grid_idx near m?
- We need: m-2 < delta * ln(n) / delta_omega < m+2
- So: exp((m-2) * delta_omega / delta) < n < exp((m+2) * delta_omega / delta)

This is a RANGE of n values! And the range is O(1) width in log-space.

## The Key Insight

For each grid point m, only a SMALL NUMBER of source points n contribute.

Because ln(n) is monotonic, and B-spline has compact support (width 4).

If grid point m corresponds to frequency ω_m = m * delta_omega,
then only n with |delta * ln(n) - ω_m| < 2 * delta_omega contribute.

That's like 4-8 source points per grid point!

## Pre-compute the Routing

Once and for all:
1. For each grid point m, compute which source points n contribute
2. Store as: sources[m] = [n1, n2, n3, ...] (variable length, but bounded)
3. Store weights[m] = [w1, w2, w3, ...] (B-spline weights)

Then the spread becomes:
```python
for m in range(num_grid):  # PARALLEL over m!
    grid[m] = sum(a_n[sources[m]] * weights[m])
```

No atomics. No collisions. Pure gather.

## But Variable Length is Bad for GPU

Each grid point has different number of sources. 2? 4? 8?

Options:
1. Pad to max (wasteful)
2. CSR format (sparse matrix)
3. Fixed stencil with masks

Actually... wait. Let's think about this more carefully.

## The Frequency Distribution

omega_n = delta * ln(n) for n = 1, 2, ..., M

ln(1) = 0
ln(2) = 0.693
ln(3) = 1.099
...
ln(M) = ln(sqrt(t/2π)) ≈ 0.5 * ln(t)

The omega_n values are NOT uniformly distributed.
- Dense near ω=0 (many small n)
- Sparse near ω_max (few large n)

Grid points near 0: many sources
Grid points near max: few sources

## Hmm, This is the NUFFT Problem

Non-uniform FFT. Sources at non-uniform frequencies.
Standard NUFFT libraries handle this with:
1. Spreading (our current approach)
2. Deconvolution (we skip this)

The spreading step IS the bottleneck in NUFFT.
Libraries use tricks like:
- Sorting sources by frequency
- Blocking for cache efficiency
- Vectorized spreading

## Back to Hollywood

What if we think of it as a CIRCUIT?

Each source n has a value a_n.
Each grid point m is a SUM node.
The connections are fixed (determined by ln(n)).

In hardware: just wires from sources to summers.
In software: we need to SIMULATE these wires.

The most efficient simulation: process by DESTINATION.

## Gather vs Scatter Redux

Scatter (current): 
- Each source writes to multiple destinations
- Write conflicts, need atomics
- Bad cache behavior (random writes)

Gather (Hollywood):
- Each destination reads from multiple sources
- No conflicts (each output independent)
- Better cache behavior (streaming reads if sorted)

## The Plan

1. Pre-compute inverse mapping: for each grid index m, list source indices n
2. Store as dense tensor with padding (or sparse CSR)
3. Gather kernel: each thread handles one grid point, sums its sources

## Complexity Check

Pre-compute: O(M) to build mapping
Gather: O(num_grid × avg_sources_per_grid) = O(num_grid × 4) = O(num_grid)

Same complexity as scatter! But:
- No atomics
- Better memory access patterns
- More parallelism (grid points independent)

## What About the Weights?

B-spline weight depends on fractional part of grid_idx.

w(n, m) = bspline(grid_idx[n] - m)

These are DIFFERENT for each (n, m) pair. Need to store them too.
Or recompute on the fly (it's just a cubic polynomial).

## Memory vs Compute Tradeoff

Option A: Pre-store weights
- Memory: O(num_grid × max_sources × 2) for (index, weight) pairs
- Compute: Just gather and sum

Option B: Compute weights on fly
- Memory: O(num_grid × max_sources) for indices only
- Compute: Gather, compute bspline, multiply, sum

Given GPU has tons of compute, Option B might be better.

## Next Steps

1. Implement the inverse mapping
2. Benchmark gather vs scatter
3. See if this breaks the 5M zeros/sec ceiling

This feels right. Hollywood Squares says: don't fight the routing, embrace it.
Make the topology explicit, then let the hardware do what it's good at.
