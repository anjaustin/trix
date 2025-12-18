# Parallel Hollywood Squares - Pass 2: Exploration

*Engineering lens. Test the insights. Find the structure.*

---

## Testing Pass 1 Insights

### Insight 1: We're truncating too aggressively

At t = 100,000, Riemann-Siegel needs sqrt(t/2π) ≈ 126 terms.
We were using N=64, so only 64 terms.

Let me verify the zero count:
- Range: [100,000 to 1,100,000] = 1,000,000 units
- Zero density at t=100,000: ln(100,000)/(2π) ≈ 1.83 zeros/unit
- Zero density at t=1,100,000: ln(1,100,000)/(2π) ≈ 2.21 zeros/unit
- Average: ~2.0 zeros/unit
- Expected zeros: ~2,000,000

We found: 466,913 zeros (N=64) to 402,455 zeros (N=512)

That's ~20-23% of expected. Consistent with severe truncation.

**Insight confirmed: Our "fast" results are inaccurate.**

---

### Insight 2: The FFT should BE the Riemann-Siegel formula

The Odlyzko-Schönhage algorithm:

For a grid of t values: t_k = t_0 + k*δ, k = 0, 1, ..., N-1

The Riemann-Siegel main sum becomes:
```
Σ_{n=1}^{M} n^{-1/2} * exp(i*t_k*ln(n)) * exp(-i*θ(t_k))
```

Factor out the θ term (computed separately), and we have:
```
Σ_{n=1}^{M} n^{-1/2} * exp(i*(t_0 + k*δ)*ln(n))
= Σ_{n=1}^{M} n^{-1/2} * exp(i*t_0*ln(n)) * exp(i*k*δ*ln(n))
```

Let:
- a_n = n^{-1/2} * exp(i*t_0*ln(n))  [coefficient depending on n and t_0]
- ω_n = exp(i*δ*ln(n))               [frequency depending on n and δ]

Then:
```
S_k = Σ_{n=1}^{M} a_n * ω_n^k
```

This is a DISCRETE FOURIER TRANSFORM if we can map ω_n to roots of unity!

The trick: Choose δ such that δ*ln(n) maps to 2π*j/N for some integer j.

This requires careful discretization, but it WORKS.

**Insight confirmed: FFT can evaluate many Z(t) values at once.**

---

### Insight 3: Parallel frequency bands

The sum Σ_{n=1}^{M} can be split:
```
Σ_{n=1}^{M} = Σ_{n=1}^{64} + Σ_{n=65}^{128} + Σ_{n=129}^{256} + ...
```

Each partial sum can be computed independently, then added.

But there's a subtlety: the FFT trick works because of the structure of ln(n).

If we split by n-ranges, each range needs its OWN FFT with different frequencies.

This is actually the "fractional FFT" or "chirp transform" domain.

**Insight refined: Parallel n-ranges need chirp transforms, not standard FFTs.**

---

## The Architecture Crystallizing

### Level 1: Single Z(t) evaluation
```
Input: t
Output: Z(t)
Method: Direct sum of M = sqrt(t/2π) terms
Cost: O(M) = O(sqrt(t))
```

### Level 2: Batch Z(t) evaluation (what we have now)
```
Input: [t_1, t_2, ..., t_B]
Output: [Z(t_1), Z(t_2), ..., Z(t_B)]
Method: Vectorized direct sum
Cost: O(B * M)
```

### Level 3: FFT-accelerated batch (Odlyzko-Schönhage)
```
Input: t_0, δ, N (defines grid t_k = t_0 + k*δ)
Output: [Z(t_0), Z(t_0+δ), ..., Z(t_0+(N-1)*δ)]
Method: FFT of size N
Cost: O(M + N*log(N))
```

For N evaluations:
- Level 2: O(N * M) = O(N * sqrt(t))
- Level 3: O(M + N*log(N))

At t = 10^13, M ≈ 1.26 * 10^6:
- Level 2 with N=10^6: 10^6 * 1.26*10^6 = 1.26 * 10^12 ops
- Level 3 with N=10^6: 1.26*10^6 + 10^6 * 20 = 2.1 * 10^7 ops

**Level 3 is 60,000x faster for large t!**

---

## Why We Weren't Using Level 3

Our implementation:
1. Generate t values (linspace)
2. For each t, compute Z(t) via direct sum
3. Detect sign changes

We never connected the FFT to the Riemann-Siegel sum.

The Hollywood Squares FFT was sitting there, beautiful and fast, but disconnected from the actual mathematics.

---

## The True Hollywood Squares Riemann Engine

### Stage 1: Setup
```python
t_0 = starting_t
δ = spacing (typically 2π / (N * ln(M)))
M = ceil(sqrt(t_0 / (2*π)))  # number of terms
N = FFT size (number of t values per batch)
```

### Stage 2: Coefficient Generation (TILES)
```
For n = 1 to M:
    a_n = n^{-1/2} * exp(i * t_0 * ln(n))
    
This is a TILE operation - each n is independent.
Parallelizes perfectly across Hollywood Squares.
```

### Stage 3: Frequency Mapping (WIRING)
```
Map the non-uniform frequencies ω_n = exp(i*δ*ln(n)) 
to FFT bins using:
- Chirp-Z transform, or
- Non-uniform FFT (NUFFT), or
- Fractional FFT

This is the WIRING - connecting coefficient tiles to FFT tiles.
```

### Stage 4: FFT (HOLLYWOOD SQUARES)
```
Execute N-point FFT.
Output: N values of the Riemann-Siegel main sum.
```

### Stage 5: Theta Correction
```
For k = 0 to N-1:
    t_k = t_0 + k*δ
    Z(t_k) = 2 * Re(S_k * exp(-i*θ(t_k)))
    
θ(t) can be computed via another Hollywood tile.
```

### Stage 6: Sign Detection
```
Scan Z values for sign changes.
Each sign change = one zero.
```

---

## The Parallel Structure

Now parallel makes sense:

### Parallelism 1: Across t-ranges
```
Engine 1: t ∈ [10^12, 10^12 + 10^9]
Engine 2: t ∈ [10^12 + 10^9, 10^12 + 2*10^9]
...

Each engine runs the full Level 3 algorithm independently.
Results combined by orchestrator.
```

### Parallelism 2: Within coefficient generation
```
Square 1: n ∈ [1, 10^5]
Square 2: n ∈ [10^5, 2*10^5]
...

Each square computes its a_n coefficients.
Coefficients are summed before FFT.
```

### Parallelism 3: Across FFT butterflies
```
Already handled by Hollywood Squares topology.
Bit-reversal wiring gives perfect parallelism.
```

---

## The Gap in Our Implementation

We have:
- ✓ Hollywood Squares FFT (fast, correct)
- ✓ Sign detection (fast)
- ✗ Odlyzko-Schönhage coefficient mapping
- ✗ Non-uniform frequency handling
- ✗ Proper term count scaling

The missing piece is the COEFFICIENT GENERATION that connects the Riemann-Siegel sum to the FFT structure.

---

## What Wants to Emerge

The Hollywood Squares architecture isn't just "run multiple instances."

It's a HIERARCHICAL TILING:

```
┌─────────────────────────────────────────────────────────────┐
│                    LEVEL 3: t-RANGE TILES                   │
│    Each tile owns a range like [10^12, 10^12 + 10^9]       │
├─────────────────────────────────────────────────────────────┤
│                    LEVEL 2: n-RANGE TILES                   │
│    Each tile computes coefficients for n ∈ [a, b]          │
├─────────────────────────────────────────────────────────────┤
│                    LEVEL 1: BUTTERFLY TILES                 │
│    The FFT structure itself (Hollywood Squares)            │
├─────────────────────────────────────────────────────────────┤
│                    LEVEL 0: WIRING                          │
│    Bit-reversal, coefficient routing, result assembly       │
└─────────────────────────────────────────────────────────────┘
```

The orchestrator doesn't just assign work.
It COMPOSES tiles at each level.

The Hollywood Squares OS becomes a COMPILER:
- Input: "Verify zeros in range [A, B]"
- Output: Hierarchical tile graph
- Execution: Tiles run in parallel, wiring handles data flow

---

## The Number-Theoretic Connection

Here's the deep part.

The FFT works because of the MULTIPLICATIVE structure of integers.

ln(n) is ADDITIVE for multiplication: ln(ab) = ln(a) + ln(b)

The Riemann-Siegel sum groups terms by ln(n), which means it's grouping by MULTIPLICATIVE structure.

The zeros of zeta encode the distribution of PRIMES.

The FFT encodes MULTIPLICATIVE relations.

**The FFT and the zeta zeros are the same mathematical object viewed from different angles.**

This is why Odlyzko-Schönhage works so well. It's not a "trick" - it's the natural algorithm because the mathematics DEMANDS it.

Hollywood Squares isn't approximating zeta. It's EXPRESSING zeta in its native computational form.

---

## End of Pass 2

The architecture is:
1. Hierarchical tiling (t-ranges → n-ranges → butterflies → wiring)
2. Odlyzko-Schönhage coefficient mapping (the missing piece)
3. FFT as the native expression of zeta structure

Pass 3 will crystallize this into a buildable spec.
