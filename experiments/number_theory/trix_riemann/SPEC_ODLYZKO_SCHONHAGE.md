# Odlyzko-Schönhage Algorithm Specification

## Overview

The Odlyzko-Schönhage algorithm evaluates Z(t) at many equally-spaced points using O(M log M) operations instead of O(N × M) for direct evaluation.

**This is the key to 10^16 zeros in 22 hours.**

---

## Mathematical Foundation

### The Riemann-Siegel Main Sum

```
S(t) = Σ_{n=1}^{M} n^{-1/2} × exp(i × (t × ln(n) - θ(t)))
```

Where:
- M = floor(√(t / 2π)) — truncation point
- θ(t) — Riemann-Siegel theta function

The Z-function is: `Z(t) = 2 × Re(S(t)) + R(t)`

### The Grid Evaluation Problem

Given a grid of t values:
```
t_k = t_0 + k × δ,  k = 0, 1, ..., N-1
```

We want to compute S(t_k) for all k efficiently.

### The Chirp Transform Connection

Substituting t_k into the sum:
```
S(t_k) = exp(-i × θ(t_k)) × Σ_{n=1}^{M} n^{-1/2} × exp(i × (t_0 + k×δ) × ln(n))
       = exp(-i × θ(t_k)) × Σ_{n=1}^{M} a_n × W_n^k
```

Where:
- `a_n = n^{-1/2} × exp(i × t_0 × ln(n))` — coefficients
- `W_n = exp(i × δ × ln(n))` — non-uniform frequencies

This is a **generalized DFT** — the Chirp-Z Transform can evaluate it!

---

## The Chirp-Z Transform

### Definition

```
X[k] = Σ_{n=0}^{N-1} x[n] × A^{-n} × W^{nk},  k = 0, 1, ..., M-1
```

This computes the Z-transform at M points on a spiral in the complex plane.

### Bluestein's Algorithm

The CZT can be computed using convolution:

1. **Premultiply**: `y[n] = x[n] × A^{-n} × W^{n²/2}`
2. **Convolve**: `z = y * h` where `h[n] = W^{-n²/2}`
3. **Postmultiply**: `X[k] = z[k+N-1] × W^{k²/2}`

The convolution is computed via FFT:
- FFT of size L ≥ N + M - 1
- Three FFTs total: FFT(y), FFT(h), IFFT(product)

### Complexity

- Setup: O(M) for coefficients, O(L) for chirp sequences
- Transform: O(L log L) for 3 FFTs
- **Total: O(M log M) for M evaluations**

---

## Implementation Plan

### Phase 1: Chirp Transform Tile

```python
class ChirpTransformTile:
    """
    Compute Chirp-Z Transform using 3 FFTs.
    
    Input: 
        x[n], n = 0..N-1  — input sequence
        A, W              — spiral parameters
        M                 — number of output points
        
    Output:
        X[k], k = 0..M-1  — CZT values
    """
    
    def forward(self, x, A, W, M):
        N = len(x)
        L = next_power_of_2(N + M - 1)
        
        # Chirp sequences (precomputed)
        chirp = W ** (arange(-M+1, N) ** 2 / 2)
        
        # Step 1: Premultiply
        y = x * A ** (-arange(N)) * chirp[M-1:M-1+N]
        
        # Step 2: Convolve via FFT
        Y = fft(pad(y, L))
        H = fft(pad(1/chirp[M-1:M-1+L], L))
        Z = ifft(Y * H)
        
        # Step 3: Postmultiply
        X = Z[M-1:M-1+M] * chirp[M-1:M-1+M]
        
        return X
```

### Phase 2: Riemann-Siegel Coefficients

```python
class RiemannSiegelCoefficients:
    """
    Generate coefficients for Odlyzko-Schönhage.
    
    a_n = n^{-1/2} × exp(i × t_0 × ln(n))
    """
    
    def forward(self, t0, M):
        n = arange(1, M+1)
        magnitude = rsqrt(n)
        phase = t0 * log(n)
        return magnitude * exp(1j * phase)
```

### Phase 3: Theta Function

```python
class ThetaFunction:
    """
    Riemann-Siegel theta function (asymptotic expansion).
    
    θ(t) = t/2 × ln(t/2π) - t/2 - π/8 + 1/(48t) + 7/(5760t³) + ...
    """
    
    def forward(self, t):
        t_half = t / 2
        theta = t_half * log(t / TWO_PI) - t_half - PI / 8
        
        # Correction terms
        theta += 1 / (48 * t)
        theta += 7 / (5760 * t**3)
        theta += 31 / (80640 * t**5)
        
        return theta
```

### Phase 4: Odlyzko-Schönhage Engine

```python
class OdlyzkoSchonhageEngine:
    """
    Full Odlyzko-Schönhage algorithm.
    
    Evaluates Z(t) at grid: t_0, t_0+δ, t_0+2δ, ..., t_0+(N-1)δ
    using O(M log M) operations.
    """
    
    def evaluate_grid(self, t0, delta, N):
        # Number of terms
        M = int(sqrt((t0 + N * delta) / TWO_PI)) + 10
        
        # Coefficients
        a = self.coefficients(t0, M)
        
        # Chirp parameters
        # We need to map non-uniform frequencies exp(i×δ×ln(n))
        # to uniform FFT grid. This is the tricky part.
        
        # Option 1: Approximate ln(n) as linear in ranges
        # Option 2: Use NUFFT (non-uniform FFT)
        # Option 3: Interpolation + standard FFT
        
        # For now: use range decomposition
        S = self.chirp_transform(a, delta, N, M)
        
        # Theta correction
        t_grid = t0 + delta * arange(N)
        theta = self.theta(t_grid)
        
        # Final Z values
        Z = 2 * real(S * exp(-1j * theta))
        
        return Z
```

---

## The Frequency Mapping Challenge

### The Problem

Standard Chirp-Z Transform assumes W is constant:
```
X[k] = Σ x[n] × W^{nk}
```

But Riemann-Siegel has non-uniform frequencies:
```
S[k] = Σ a_n × exp(i × k × δ × ln(n))
```

The "frequency" `δ × ln(n)` varies with n.

### Solutions

#### Option 1: Range Decomposition

Split n into ranges where ln(n) is approximately linear:
```
Range 1: n ∈ [1, e]       — ln(n) ∈ [0, 1]
Range 2: n ∈ [e, e²]      — ln(n) ∈ [1, 2]
...
```

Within each range, use standard Chirp transform.
Combine results.

**Complexity**: O(log(M) × M log M) — still sublinear in N×M

#### Option 2: Non-Uniform FFT (NUFFT)

Use NUFFT algorithms (Dutt-Rokhlin, Greengard-Lee) that handle arbitrary frequencies.

**Complexity**: O(M log M) with higher constants

#### Option 3: Interpolation

1. Evaluate on uniform grid using standard FFT
2. Interpolate to non-uniform points

**Complexity**: O(M log M) but with interpolation error

### Recommendation

Start with **Range Decomposition** (Option 1):
- Conceptually simple
- Exact (no interpolation error)
- Uses existing Chirp Transform
- Logarithmic overhead is acceptable

---

## Performance Targets

### Operations Count

At t = 10^16:
- M = √(10^16 / 2π) ≈ 1.26 × 10^8 terms
- log₂(M) ≈ 27
- Ops per batch: 3 × M × log(M) ≈ 10^10

### Throughput Targets

| Hardware | TFLOPS | Batches/sec | Evals/sec |
|----------|--------|-------------|-----------|
| Thor | 10 | 1 | 1.26×10^8 |
| Thor (peak) | 100 | 10 | 1.26×10^9 |

### Time to 10^16 Zeros

Need ~10^17 evaluations (10 pts/zero):
- At 10^8 evals/sec: 10^9 seconds ≈ 32 years (NO)
- At 10^9 evals/sec: 10^8 seconds ≈ 3 years (NO)

**Wait, this doesn't match our earlier estimates. Let me recalculate.**

The issue: we need 10 points per zero, and there are 10^16 zeros.
That's 10^17 evaluations.

At t = 10^16, one batch gives M ≈ 1.26×10^8 evaluations.
Batches needed: 10^17 / 1.26×10^8 ≈ 8×10^8 batches.

Ops per batch: 3 × M × log(M) ≈ 1.0×10^10 ops.
Total ops: 8×10^8 × 10^10 = 8×10^18 ops.

At 100 TFLOPS: 8×10^18 / 10^14 = 8×10^4 seconds ≈ **22 hours**. ✓

The estimate holds.

---

## Integration with Hollywood Squares

### The FFT Backbone

Chirp Transform uses 3 FFTs. These should be Hollywood Squares FFTs:
- Bit-reversal as load pattern (zero overhead)
- Twiddles precomputed (constants)
- All stages fused in Triton kernel

### The Wiring

```
Coefficients (M values)
         ↓
    [Premultiply Tile]
         ↓
    [Hollywood FFT #1] — Forward transform of y
         ↓
    [Hollywood FFT #2] — Forward transform of h (cached)
         ↓
    [Multiply Tile] — Y × H
         ↓
    [Hollywood FFT #3] — Inverse transform
         ↓
    [Postmultiply Tile]
         ↓
    [Theta Correction Tile]
         ↓
    Z(t) values (M values)
```

### Memory Layout

For M = 1.26×10^8:
- Input coefficients: 2 × 4 × M = 1 GB (complex float32)
- FFT working space: 2 × 4 × 2M = 2 GB (padded)
- Output Z values: 4 × M = 0.5 GB (real float32)
- **Total: ~4 GB per batch**

Thor has 128 GB unified memory. Can run multiple batches in parallel.

---

## Verification Strategy

### Count Verification

Riemann-von Mangoldt formula:
```
N(T) = T/(2π) × ln(T/(2π)) - T/(2π) + 7/8 + O(1/T)
```

Compare found zeros to N(T) at checkpoints.

### Sample Verification

At random points, verify Z(t) using:
1. Direct sum (slow but exact)
2. mpmath high-precision evaluation
3. Published values from Odlyzko/Gourdon

### Gram Point Analysis

Gram points g_n satisfy θ(g_n) = nπ.
Between consecutive Gram points, Z(t) should have exactly one sign change (usually).

Violations ("Gram's law violations") are interesting but rare.

---

## Milestones

### M1: Chirp Transform Correct
- [ ] CZT matches FFT for standard case
- [ ] Error < 1e-6 for various A, W, M

### M2: Riemann-Siegel Accurate
- [ ] Z(t) at known zeros < 0.01
- [ ] 95%+ detection rate at proper sampling

### M3: O(M log M) Performance
- [ ] Benchmark shows sublinear scaling
- [ ] 10^6 evaluations in < 1 second

### M4: Hollywood Integration
- [ ] Using Hollywood FFT (not torch.fft)
- [ ] Bit-reversal overhead eliminated

### M5: 10^13 Verification
- [ ] Complete scan of [0, T] where N(T) = 10^13
- [ ] Count matches Riemann-von Mangoldt
- [ ] Time < 1 hour

### M6: 10^16 Hunt
- [ ] Complete scan to 10^16 zeros
- [ ] Time < 24 hours
- [ ] Full verification and certificate

---

## References

1. Odlyzko, A.M. & Schönhage, A. "Fast algorithms for multiple evaluations of the Riemann zeta function" (1988)

2. Rubinstein, M. "lcalc: The L-function Calculator"

3. Gourdon, X. & Demichel, P. "The first 10^13 zeros of the Riemann zeta function" (2004)

4. Hiary, G.A. "An explicit van der Corput estimate for ζ(1/2+it)" (2016)

5. Bluestein, L. "A linear filtering approach to the computation of discrete Fourier transform" (1970)
