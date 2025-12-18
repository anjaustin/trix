# Parallel Hollywood Squares - Pass 1: Raw Thoughts

*Stream of consciousness. No filter. Let it flow.*

---

## The Numbers That Triggered This

```
Single instance: 24M zeros/sec → 4.9 days for 10^13
Parallel (20x):  480M zeros/sec → 6 hours for 10^13
```

Wait. Stop. Something's off.

We just casually said "20 instances" like they're independent processes. But they're not independent. They share:
- The same GPU memory
- The same memory bandwidth
- The same compute units

If 1 instance saturates the GPU, 20 instances don't give 20x. They give... 1x with more overhead.

So why does parallelism work at all?

---

## What Is Actually Parallel?

The zeros are independent. Range [0, 10^12] doesn't depend on [10^12, 2×10^12].

But the COMPUTATION for each zero isn't independent:
- They all need FFT
- They all need twiddles
- They all need theta computation
- They all need sign detection

The parallelism isn't in the instances. It's in the BATCH SIZE.

When we went from 1M points to 10M points, we got:
- 1M: 19.2M zeros/sec
- 10M: 23.8M zeros/sec

The improvement came from BATCHING, not from multiple instances.

---

## The Real Question

What's the limiting factor?

1. **Compute bound**: GPU cores fully utilized
2. **Memory bound**: Bandwidth saturated
3. **Latency bound**: Waiting on something

At N=64 with 10M points:
- 10M * 64 * 2 * 4 bytes = 5.12 GB of data
- Processing in 0.22s = 23 GB/s

Thor memory bandwidth: 204.8 GB/s

We're at ~11% of theoretical bandwidth. There's room.

---

## Why N=64 Beats N=256?

N=64:  19.2M zeros/sec
N=256: 5.0M zeros/sec

4x smaller FFT → 4x more throughput?

No. The FFT isn't the bottleneck. Look at what else scales:

At N=64:
- N_terms for Riemann-Siegel: sqrt(t/2π) ≈ 126 terms at t=100,000
- But we capped at N_max = min(N, 2000) = 64 terms

We're TRUNCATING the Riemann-Siegel sum too early!

N=64 is fast because it's doing LESS WORK, not because FFT is faster.

---

## The Accuracy Question

Z(14.134725) = -0.212397 (should be ~0)

It's NOT zero. The first zero of zeta is at t ≈ 14.134725, and we're getting -0.21.

That's a 21% error at a KNOWN zero.

Are we detecting real zeros or noise?

Let me think... at t = 100,000:
- True zero density: ~ln(t)/(2π) zeros per unit t
- At t=100k: about 1.83 zeros per unit t
- Over range [100k, 1.1M]: about 1.83M zeros expected

We found 466,913 zeros in that range. That's ~25% of expected.

We're MISSING 75% of the zeros because our truncation is too aggressive.

---

## The Tradeoff

Speed vs Accuracy

N=64:  Fast, but truncated at 64 terms → missing zeros
N=256: Slower, but 256 terms → more accurate
N=1024: Slowest, but 1024 terms → most accurate

The "24M zeros/sec" is a lie if we're missing most of them.

---

## What Actually Matters for Riemann Hypothesis?

We don't need to FIND every zero. We need to VERIFY that:
1. All zeros in a range have real part = 1/2
2. The count matches the expected count (Riemann-von Mangoldt formula)

For verification:
- N(T) = (T/2π) * ln(T/2π) - T/2π + O(ln T)
- Compare found zeros to N(T)

If our count matches N(T), we've verified the range even if we missed individual zeros.

But wait - if we're missing 75%, the count WON'T match.

---

## The Emergence

I think I see it now.

The bottleneck isn't FFT. It isn't parallelism. It isn't memory bandwidth.

**The bottleneck is the number of terms in the Riemann-Siegel sum.**

For large t, we need sqrt(t/2π) terms.
At t = 10^13, that's sqrt(10^13 / 6.28) ≈ 1.26 × 10^6 terms.

1.26 MILLION terms per Z(t) evaluation.

No wonder N=64 is "fast" - it's only computing 64 terms instead of a million.

---

## The Hollywood Squares Insight Returns

Wait.

The Riemann-Siegel sum is:
Z(t) = 2 * Σ n^(-1/2) * cos(θ(t) - t*ln(n))

This is a SUM of COSINES with different frequencies.

That's... that's an INVERSE FFT problem!

If we can express the sum as an FFT, we can evaluate MANY Z(t) values simultaneously with a single FFT.

This is exactly what Odlyzko-Schönhage does!

---

## The Real Architecture

```
      t values: [t₀, t₁, t₂, ..., t_N]
                       ↓
         Coefficient Generation (n^(-1/2) terms)
                       ↓
              SINGLE FFT (size N)
                       ↓
      Z values: [Z(t₀), Z(t₁), Z(t₂), ..., Z(t_N)]
```

One FFT gives us N evaluations of Z(t)!

But we weren't using the FFT for this. We were:
1. Generating t values
2. Computing Z(t) INDIVIDUALLY for each t
3. Using FFT only for... nothing useful?

The FFT should BE the Riemann-Siegel sum, not a separate component.

---

## The Summit Beyond the Summit

The Hollywood Squares FFT isn't just for "some computation".

It IS the Riemann-Siegel formula.

The wiring pattern of the FFT corresponds to the frequency structure of the zeta zeros.

**The topology of the FFT IS the topology of the zeta function.**

This is why Odlyzko-Schönhage works. The FFT naturally encodes the multiplicative structure of the integers (via n = p₁^a₁ * p₂^a₂ * ...).

---

## What Wants to Emerge

The parallel instances aren't parallel PROCESSES.

They're parallel FREQUENCY BANDS.

Like a graphic equalizer, each "Hollywood Square" handles a different frequency range of the zeta function.

Low frequencies: contribution from small n (1, 2, 3, ...)
High frequencies: contribution from large n

The orchestrator doesn't assign t-ranges.
It assigns FREQUENCY RANGES.

And the final Z(t) is assembled by summing the contributions.

This is the ACTUAL Hollywood Squares architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Z(t) ASSEMBLY                            │
├─────────────────────────────────────────────────────────────┤
│  Square 1     Square 2     Square 3     Square N            │
│  n ∈ [1,64]   n ∈ [65,128] n ∈ [129,192] ...               │
│  Low freq     Mid freq     High freq    ...                 │
├─────────────────────────────────────────────────────────────┤
│                    SHARED t VALUES                          │
└─────────────────────────────────────────────────────────────┘
```

Each square computes its frequency band's contribution.
The orchestrator sums them.
THAT'S true parallelism.

---

## End of Pass 1

Raw insight: The FFT isn't a tool we use. It IS the Riemann-Siegel formula.

Parallel Hollywood Squares = parallel frequency bands, not parallel t-ranges.

Something deeper is here. Need Pass 2 to explore.
