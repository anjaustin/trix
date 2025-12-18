# Hollywood Squares OS × NUFFT: Layered Architecture

## Raw Stream of Consciousness

What IS Hollywood Squares OS at its core?

"Routing is free."

In a crossbar switch, connecting input 5 to output 3 costs NOTHING.
It's just a wire. The topology IS the computation.

Current thinking: we COMPUTE things, then ROUTE results.
Hollywood thinking: we DEFINE TOPOLOGY, then data flows through.

## The Riemann Pipeline

```
n=1..M  →  [Coefficients]  →  [Spread]  →  [FFT]  →  [Theta]  →  Z(t)
                a_n            grid        G        correction   output
```

Each arrow is "routing" - moving data from one place to another.
Each box is "computation" - transforming data.

But what if the ROUTING is the expensive part?

## Where's The Routing?

1. **Coefficients**: n → a_n. No routing, just compute.

2. **Spread**: a_n → grid. THIS IS ALL ROUTING!
   - Source n goes to grid[floor(grid_idx[n]) + {-1,0,1,2}]
   - The B-spline weight is just "how strong is this wire"
   - It's a sparse matrix multiply: grid = S @ a_n

3. **FFT**: grid → G. This IS routing!
   - Butterfly network is pure permutation + twiddle multiply
   - The permutations are TOPOLOGY
   - The twiddles are just "wire weights"

4. **Theta**: G → Z. No routing, just pointwise computation.

## The Realization

Steps 2 and 3 are BOTH routing operations!

Spreading: sparse routing (each input → 4 outputs)
FFT: structured routing (butterfly pattern)

What if we MERGE them?

## Merged Routing

Currently:
```
a_n → [Spread] → grid → [FFT] → G
```

Both are linear operations:
- Spread: grid = S @ a_n  (sparse matrix)
- FFT: G = F @ grid       (FFT matrix)

Combined: G = F @ S @ a_n = (F @ S) @ a_n

The matrix (F @ S) is the COMPLETE TOPOLOGY from sources to outputs!

## Pre-compute The Topology

(F @ S) is a matrix of size (num_evals × M).
For M=12625, num_evals=101000, that's 1.3B complex numbers.
At 16 bytes each = 20 GB. Too big to store.

But wait - do we need the full matrix?

## Sparse Structure

S is sparse: only 4 non-zeros per column (each source → 4 grid points)
F is dense but structured: FFT

F @ S is... still pretty dense because FFT mixes everything.

Hmm, this direct approach doesn't work.

## Different Angle: Topology Per Batch

What if we think about BATCHING?

For a given t0, the topology is fixed:
- grid_idx[n] = delta * ln(n) / delta_omega (fixed)
- Spread pattern (fixed)
- FFT butterflies (fixed)

If we process 1000 batches with the same t0, we reuse the topology 1000x.

## Even Better: Topology Across t0

As t0 changes, what actually changes?

grid_idx[n] = delta * ln(n) / delta_omega

delta = 1 / (density * 10) where density = ln(t0) / (2π)

So delta ∝ 1/ln(t0), which changes slowly!

For t0 from 10^9 to 1.001×10^9:
- ln(10^9) = 20.72
- ln(1.001×10^9) = 20.72 + 0.001 = 20.721

Almost no change! The topology is STABLE across nearby t0 values.

## The Hollywood Insight

We can compute one "Hollywood Topology" and reuse it for a HUGE range of t0!

1. Pick representative t0
2. Compute spread indices, FFT plan
3. Bake into fixed arrays
4. Run millions of batches through the same topology

The only thing that changes: coefficient phases exp(i * t0 * ln(n))

## Implementation Sketch

```python
class HollywoodRiemannEngine:
    def __init__(self, t0_center, t_range, num_evals):
        # Pre-compute topology ONCE
        self.spread_indices = compute_spread_indices(...)
        self.spread_weights = compute_spread_weights(...)
        self.fft_plan = create_fft_plan(...)
        
        # Pre-compute things that don't change
        self.ln_n = torch.log(torch.arange(1, M+1))
        self.rsqrt_n = torch.rsqrt(torch.arange(1, M+1))
    
    def evaluate_batch(self, t0_batch):
        # Only compute coefficient phases (changes with t0)
        phases = t0_batch.unsqueeze(-1) * self.ln_n  # (batch, M)
        a_n = self.rsqrt_n * torch.exp(1j * phases)
        
        # Apply pre-computed topology (FAST - just indexing)
        grid = spread_with_fixed_topology(a_n, self.spread_indices, self.spread_weights)
        
        # FFT with pre-computed plan
        G = self.fft_plan.execute(grid)
        
        # Theta correction
        Z = theta_correct(G, t0_batch)
        
        return Z
```

## The Key Optimization

`spread_with_fixed_topology` is NOT scatter_add anymore!

It's a GATHER with pre-computed indices:
```python
# Pre-computed: for each (grid_point, offset), which source contributes
# source_idx[m, k] = which n contributes to grid[m] with offset k
# source_weight[m, k] = what weight

grid[m] = sum_k(a_n[source_idx[m,k]] * source_weight[m,k])
```

This is a batched gather-multiply-reduce. No atomics. No scatter.
Pure coalesced memory access.

## But Wait, We Already Tried This

In the gather exploration, we found it was slower because of intermediate tensors.

The difference with Hollywood layering:
1. Pre-compute EVERYTHING into dense tensors
2. Use matrix multiply instead of scatter/gather
3. Fuse with FFT

## Actually, This Is Just Sparse Matrix × Dense Vector

Spreading is: grid = S @ a_n where S is sparse (4 entries per column)

Sparse matrix multiply is well-optimized!

```python
# Build sparse matrix S once
# S[m, n] = bspline_weight(grid_idx[n] - m) if |grid_idx[n] - m| < 2, else 0

grid = torch.sparse.mm(S, a_n)  # or cusparse
```

Then FFT as usual.

## The Layering

Layer 1: Input preparation (coefficients)
Layer 2: Hollywood Routing (sparse matrix = topology)
Layer 3: FFT (another routing layer, already optimized)
Layer 4: Output correction (theta, pointwise)

The Hollywood insight: Layers 2 and 3 are BOTH routing.
Pre-compute layer 2's topology, use cuFFT's pre-computed plan for layer 3.

## What About Fusing 2 and 3?

FFT of sparse input is a known problem.
Sparse FFT algorithms exist but are complex.

Simpler: just make layer 2 as fast as possible with sparse matrix multiply.

## Next Steps

1. Build the sparse spreading matrix S
2. Benchmark sparse mm vs current scatter
3. If faster, integrate into pipeline
4. Explore fusing with FFT via custom Triton kernel
