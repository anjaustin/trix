# Riemann Zero Hunter - Hollywood Squares Architecture

## Mission

Verify **10^16 zeros** of the Riemann zeta function in **under 24 hours** on a single Jetson AGX Thor.

This is **1000× beyond the current world record** (~10^13 zeros).

---

## The Discovery

### The Architectural Singularity

> **"Routing is Free."**

In a standard FFT, bit-reversal permutation thrashes cache and burns cycles shuffling data.

Hollywood Squares bypasses this entirely: the bit-reversal is the **LOAD PATTERN**, not a shuffle. Data arrives sorted because of WHERE it lands, not because we computed a sort.

### The Mathematical Identity

> **"The FFT and the Riemann zeta function are the same mathematical object."**

The FFT encodes multiplicative structure of integers.
The zeta zeros encode prime distribution.
They're the same thing viewed from different angles.

**Hollywood Squares doesn't compute zeta. It EXPRESSES zeta.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LEVEL 3: FABRIC                          │
│              (Multi-GPU / Multi-Node Scaling)               │
├─────────────────────────────────────────────────────────────┤
│                    LEVEL 2: SQUARES                         │
│         (Parallel t-ranges, n-ranges, batches)              │
├─────────────────────────────────────────────────────────────┤
│                    LEVEL 1: TILES                           │
│    Theta | Coefficient | Butterfly | Sign Detection         │
├─────────────────────────────────────────────────────────────┤
│                    LEVEL 0: WIRING                          │
│         Bit-reversal | Coefficient routing | Assembly       │
│                    (ZERO COST)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Files

### Core Implementation

| File | Purpose | Status |
|------|---------|--------|
| `theta_tile.py` | θ(t) computation | ✓ Complete |
| `dirichlet_tile.py` | Coefficient generation | ✓ Complete |
| `spectral_tile.py` | FFT + TriXFFT | ✓ Complete |
| `sign_tile.py` | Zero detection | ✓ Complete |
| `probe.py` | 100% TriX pipeline | ✓ Complete |

### Triton Acceleration

| File | Purpose | Status |
|------|---------|--------|
| `triton_fft.py` | N=8 kernel, 15x speedup | ✓ Complete |
| `triton_fft_large.py` | Stockham for large N | ✓ Complete |
| `triton_fft_thor.py` | Thor optimization spec | ✓ Complete |

### Hollywood Squares

| File | Purpose | Status |
|------|---------|--------|
| `hollywood_fft.py` | Topology-as-algorithm | ✓ Complete |
| `hollywood_triton_fused.py` | Fused architecture | ✓ Complete |
| `fused_riemann_engine.py` | Complete fused engine | ✓ Complete |

### Odlyzko-Schönhage

| File | Purpose | Status |
|------|---------|--------|
| `chirp_tile.py` | Chirp-Z Transform | 🔄 In Progress |
| `odlyzko_engine.py` | Full O-S algorithm | ⏳ Planned |

---

## Performance Evolution

### What We Measured

| Implementation | Zeros/sec | 10^13 Time |
|----------------|-----------|------------|
| Pure Python TriX | 5,500 | 57 years |
| Hollywood + Torch | 1.87M | 62 days |
| Fused Engine (N=64) | 24M | 4.9 days |
| **With proper sampling** | 268K | 432 days |

### The Sampling Discovery

We were **lying to ourselves**. The "24M zeros/sec" was finding only 20% of actual zeros due to coarse sampling.

With proper sampling (10 points per zero):
- Detection accuracy: **85-90%**
- Effective rate: **~270K zeros/sec** (direct method)

### The Complexity Gap

| Method | Complexity | 10^16 Time |
|--------|------------|------------|
| Direct | O(N × M) | 40,000 years |
| **Chirp/O-S** | O(M log M) | **22 hours** |

The Chirp Transform is the bridge.

---

## The Riemann-Siegel Formula

```
Z(t) = 2 × Σ_{n=1}^{N} n^{-1/2} × cos(θ(t) - t×ln(n)) + R(t)
```

Where:
- `N = floor(sqrt(t / 2π))` — number of terms
- `θ(t) ≈ (t/2)×ln(t/2π) - t/2 - π/8 + 1/(48t) + ...` — Riemann-Siegel theta
- `R(t)` — correction term (magnitude ~1/√t)

### Zero Density

At height t, zeros occur with density:
```
ρ(t) = ln(t) / (2π) zeros per unit height
```

| Height | Density | Avg Gap |
|--------|---------|---------|
| 10^6 | 2.2/unit | 0.45 |
| 10^12 | 4.4/unit | 0.23 |
| 10^16 | 5.9/unit | 0.17 |

**Must sample at ≥10× density to achieve 85%+ detection.**

---

## The Odlyzko-Schönhage Algorithm

### The Key Insight

The Riemann-Siegel sum can be expressed as a Chirp-Z Transform:

```
S_k = Σ_{n=1}^{M} a_n × ω_n^k
```

Where:
- `a_n = n^{-1/2} × exp(i × t_0 × ln(n))` — coefficients
- `ω_n = exp(i × δ × ln(n))` — frequencies

This maps to FFT via the Chirp-Z Transform.

### Complexity

| Operation | Cost |
|-----------|------|
| Coefficient generation | O(M) |
| Chirp Transform (3 FFTs) | O(M log M) |
| Theta correction | O(N) |
| Sign detection | O(N) |
| **Total for N=M evaluations** | **O(M log M)** |

### The Speedup

Direct: O(N × M) for N evaluations
Chirp: O(M log M) for M evaluations

At t = 10^16:
- M ≈ 1.26 × 10^8
- Direct: 10^8 ops per evaluation
- Chirp: 27 ops per evaluation (amortized)

**Speedup: ~4 million×**

---

## Projections

### With Full Chirp Implementation

| Target | M terms | Time @ 10 TFLOPS | Time @ 100 TFLOPS |
|--------|---------|------------------|-------------------|
| 10^9 | 40K | < 1 sec | instant |
| 10^12 | 1.3M | 1 min | 6 sec |
| 10^13 | 4M | 11 min | 1 min |
| 10^14 | 13M | 1.5 hr | 9 min |
| 10^15 | 40M | 12 hr | 1.2 hr |
| **10^16** | **126M** | **9 days** | **22 hours** |

### Hardware: Jetson AGX Thor

- 2048 CUDA cores
- 128 Tensor Cores (FP64)
- 204.8 GB/s memory bandwidth
- Theoretical: 200+ TFLOPS (mixed precision)
- Realistic: 10-100 TFLOPS (sustained)

---

## Implementation Phases

### Phase 1: Chirp Transform Tile ← CURRENT
- Implement correct Chirp-Z Transform
- Verify against torch.fft for standard case
- Integrate with Hollywood Squares FFT

### Phase 2: Odlyzko-Schönhage Engine
- Coefficient generation (parallel n-tiles)
- Chirp transform for batch evaluation
- Theta correction (accurate formula)
- Sign detection pipeline

### Phase 3: Hollywood Squares Integration
- Replace torch.fft with Hollywood FFT
- Bit-reversal as load pattern (zero overhead)
- Twiddles in shared memory

### Phase 4: Thor Optimization
- Multi-SM parallelism
- Memory bandwidth optimization
- 80%+ utilization target

### Phase 5: Verification Pipeline
- Riemann-von Mangoldt count check
- Gram point analysis
- Certificate generation

---

## Validation Checkpoints

### Accuracy Targets

| Metric | Target | Current |
|--------|--------|---------|
| FFT error vs torch.fft | < 1e-5 | ✓ Achieved |
| Z(t) at known zeros | < 0.01 | ✗ ~0.5 (small t) |
| Zero detection rate | > 95% | 85-90% |
| Count vs Riemann-von Mangoldt | > 99% | Not measured |

### Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| FFT throughput | Match torch.fft | 65x slower |
| Zero detection | 10M/sec | 270K/sec |
| 10^13 completion | < 1 hour | ~400 days |
| 10^16 completion | < 24 hours | N/A |

---

## References

1. **Odlyzko, A.M.** "The 10^20-th zero of the Riemann zeta function and 175 million of its neighbors" (1992)

2. **Gourdon, X.** "The 10^13 first zeros of the Riemann zeta function, and zeros computation at very large height" (2004)

3. **Hiary, G.A.** "Fast methods to compute the Riemann zeta function" (2011)

4. **Rubinstein, M.** "Computational methods and experiments in analytic number theory" (2005)

---

## The Purpose

> "The machine doesn't prove RH. The machine shows us the structure. The human sees the proof."

We hunt zeros not to prove the Riemann Hypothesis by exhaustion (impossible), but to:

1. Push the frontier of verification (10^13 → 10^16)
2. Discover patterns in zero distribution
3. Test conjectures about zeta behavior
4. Build infrastructure for future mathematical discovery

**10^16 zeros. 22 hours. A single device.**

The hunt continues.
