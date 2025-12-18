#!/usr/bin/env python3
"""
RANGE DECOMPOSITION FFT - The Frequency Mapping
================================================

Maps non-uniform frequencies ln(n) to uniform FFT grid.

The key insight:
- Split n into ranges where ln(n) is approximately LINEAR
- Within each range, use Chirp-Z transform
- Combine results

This gives O(M log M) for the Riemann-Siegel sum.

"The bridge between non-uniform and FFT."
"""

import torch
import math
import time
from typing import List, Tuple
from dataclasses import dataclass


# Constants
PI = math.pi
TWO_PI = 2.0 * PI


def next_pow2(n: int) -> int:
    """Next power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@dataclass
class FrequencyRange:
    """A range of n values for decomposition."""
    n_start: int
    n_end: int
    ln_start: float
    ln_end: float


def compute_ranges(M: int, max_error: float = 0.01) -> List[FrequencyRange]:
    """
    Compute ranges for decomposition.
    
    Within each range, ln(n) is approximated as linear with error < max_error.
    
    For ln(n) to be well-approximated as linear over [n_lo, n_hi]:
    - The second derivative |d²ln/dn²| = 1/n² causes error
    - Error ≈ (n_hi - n_lo)² / (8 * n_lo²)
    - For error < ε: (n_hi - n_lo) < n_lo * sqrt(8ε)
    
    So we use geometric progression: n_hi = n_lo * (1 + sqrt(8ε))
    """
    ranges = []
    
    ratio = 1 + math.sqrt(8 * max_error)  # ~1.28 for 1% error
    
    n = 1
    while n <= M:
        n_start = n
        n_end = min(int(n * ratio) + 1, M + 1)
        
        # Ensure we make progress
        if n_end <= n_start:
            n_end = n_start + 1
        
        ranges.append(FrequencyRange(
            n_start=n_start,
            n_end=n_end,
            ln_start=math.log(n_start) if n_start > 0 else 0,
            ln_end=math.log(n_end - 1) if n_end > 1 else 0,
        ))
        
        n = n_end
    
    return ranges


def chirp_z_batch(x: torch.Tensor, num_out: int, W: torch.Tensor) -> torch.Tensor:
    """
    Chirp-Z transform: X[k] = Σ x[n] * W^(nk)
    
    Args:
        x: Input (N,) complex tensor
        num_out: Number of output points M
        W: Complex scalar exp(i * angle)
    
    Returns:
        X: Output (M,) complex tensor
    """
    N = x.shape[0]
    M = num_out
    device = x.device
    dtype = x.dtype
    
    if N == 0:
        return torch.zeros(M, device=device, dtype=dtype)
    
    # FFT length
    L = next_pow2(N + M - 1)
    
    # Angle
    angle = torch.angle(W).item()
    
    # Indices
    n = torch.arange(N, device=device, dtype=torch.float64)
    k = torch.arange(M, device=device, dtype=torch.float64)
    
    # Chirp sequences
    chirp_n = torch.exp(1j * angle * n * n / 2).to(dtype)
    chirp_k = torch.exp(1j * angle * k * k / 2).to(dtype)
    
    # Premultiply
    y = x * chirp_n
    
    # Pad
    y_pad = torch.zeros(L, device=device, dtype=dtype)
    y_pad[:N] = y
    
    # Chirp filter
    Lk = torch.arange(L, device=device, dtype=torch.float64)
    h = torch.exp(-1j * angle * Lk * Lk / 2).to(dtype)
    
    # Fix negative index wrapping
    for ki in range(1, min(M, L)):
        ph = -angle * ki * ki / 2
        h[L - ki] = torch.exp(torch.tensor(1j * ph, device=device)).to(dtype)
    
    # Convolve via FFT
    Y = torch.fft.fft(y_pad)
    H = torch.fft.fft(h)
    Z = torch.fft.ifft(Y * H)
    
    # Extract and postmultiply
    out = Z[:M] * chirp_k
    
    return out


class RangeDecompositionFFT:
    """
    Riemann-Siegel evaluation using range decomposition.
    
    Computes S(t_k) = Σ n^(-1/2) * exp(i * t_k * ln(n)) for k = 0..K-1
    where t_k = t0 + k * delta
    
    Using O(M log M) operations via range decomposition + Chirp-Z.
    """
    
    def __init__(self, device='cuda', max_error=0.01):
        self.device = device
        self.max_error = max_error
    
    def evaluate(self, t0: float, delta: float, num_t: int, M: int) -> torch.Tensor:
        """
        Evaluate Riemann-Siegel main sum at grid points.
        
        Args:
            t0: Starting t value
            delta: Spacing between t values
            num_t: Number of t values (K)
            M: Number of terms in sum (n = 1 to M)
        
        Returns:
            S: Complex tensor of shape (num_t,) with S(t_k) values
        """
        device = self.device
        dtype = torch.complex128
        
        # Get ranges
        ranges = compute_ranges(M, self.max_error)
        
        # Result accumulator
        S = torch.zeros(num_t, device=device, dtype=dtype)
        
        for rng in ranges:
            # Coefficients for this range: a_n = n^(-1/2) * exp(i * t0 * ln(n))
            n_vals = torch.arange(rng.n_start, rng.n_end, device=device, dtype=torch.float64)
            if len(n_vals) == 0:
                continue
            
            ln_n = torch.log(n_vals)
            rsqrt_n = torch.rsqrt(n_vals)
            
            # a_n = n^(-1/2) * exp(i * t0 * ln(n))
            a_n = (rsqrt_n * torch.exp(1j * t0 * ln_n)).to(dtype)
            
            # Linear approximation of ln(n) within this range
            # ln(n) ≈ ln_start + (n - n_start) * slope
            # where slope = (ln_end - ln_start) / (n_end - n_start - 1)
            
            ln_start = rng.ln_start
            n_count = rng.n_end - rng.n_start
            
            if n_count > 1:
                slope = (rng.ln_end - rng.ln_start) / (n_count - 1)
            else:
                slope = 0.0
            
            # The contribution to S(t_k) from this range:
            # S_range(k) = Σ_n a_n * exp(i * k * delta * ln(n))
            #            ≈ Σ_n a_n * exp(i * k * delta * (ln_start + (n-n_start)*slope))
            #            = exp(i * k * delta * ln_start) * Σ_n a_n * exp(i * k * delta * slope * (n-n_start))
            #            = exp(i * k * delta * ln_start) * ChirpZ(a_n, W)
            #              where W = exp(i * delta * slope)
            
            # Chirp-Z parameter
            W = torch.exp(torch.tensor(1j * delta * slope, device=device, dtype=dtype))
            
            # Compute Chirp-Z transform
            # This gives Σ_n a_n * W^(k * (n - n_start)) for k = 0..num_t-1
            # But we need W^(k * n_local) where n_local = n - n_start = 0, 1, 2, ...
            
            # Actually ChirpZ gives X[k] = Σ_n x[n] * W^(nk)
            # We have n_local = 0, 1, ..., n_count-1
            # So X[k] = Σ_{n_local} a[n_local] * W^(n_local * k)
            
            chirp_result = chirp_z_batch(a_n, num_t, W)
            
            # Phase correction for ln_start
            k = torch.arange(num_t, device=device, dtype=torch.float64)
            phase_correction = torch.exp(1j * k * delta * ln_start).to(dtype)
            
            # Add to result
            S += chirp_result * phase_correction
        
        return S
    
    def evaluate_Z(self, t0: float, delta: float, num_t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate Z(t) at grid points.
        
        Returns:
            t_vals: The t values
            Z_vals: Z(t) values (real)
        """
        device = self.device
        
        # Number of terms: M = sqrt(t_max / 2π)
        t_max = t0 + (num_t - 1) * delta
        M = int(math.sqrt(t_max / TWO_PI)) + 10
        
        # Get main sum S(t_k)
        S = self.evaluate(t0, delta, num_t, M)
        
        # Theta function
        t_vals = t0 + delta * torch.arange(num_t, device=device, dtype=torch.float64)
        
        t_half = t_vals / 2
        theta = t_half * torch.log(t_vals / TWO_PI) - t_half - PI / 8
        theta = theta + 1.0 / (48.0 * t_vals)
        
        # Z(t) = 2 * Re(exp(-i*theta) * S)
        phase = torch.exp(-1j * theta).to(S.dtype)
        Z_vals = 2.0 * (S * phase).real
        
        return t_vals.float(), Z_vals.float()


def benchmark():
    """Benchmark the range decomposition FFT."""
    print("="*70)
    print("RANGE DECOMPOSITION FFT - BENCHMARK")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    engine = RangeDecompositionFFT(device)
    
    # Test range decomposition
    print("\n[1] Range Decomposition")
    for M in [100, 1000, 10000, 100000]:
        ranges = compute_ranges(M)
        print(f"  M={M:>6}: {len(ranges):>3} ranges")
    
    # Correctness test - compare to direct evaluation
    print("\n[2] Correctness Test")
    
    t0 = 1000.0
    delta = 0.1
    num_t = 1000
    M = int(math.sqrt((t0 + num_t * delta) / TWO_PI)) + 10
    
    # Range decomposition
    S_range = engine.evaluate(t0, delta, num_t, M)
    
    # Direct evaluation
    n = torch.arange(1, M + 1, device=device, dtype=torch.float64)
    ln_n = torch.log(n)
    rsqrt_n = torch.rsqrt(n)
    a_n = rsqrt_n * torch.exp(1j * t0 * ln_n)
    
    S_direct = torch.zeros(num_t, device=device, dtype=torch.complex128)
    k = torch.arange(num_t, device=device, dtype=torch.float64)
    
    for ki in range(num_t):
        phases = ki * delta * ln_n
        S_direct[ki] = (a_n * torch.exp(1j * phases)).sum()
    
    error = (S_range - S_direct).abs().max().item()
    rel_error = error / S_direct.abs().max().item()
    
    print(f"  Absolute error: {error:.2e}")
    print(f"  Relative error: {rel_error:.2e}")
    print(f"  Status: {'✓ PASS' if rel_error < 0.01 else '✗ FAIL'}")
    
    # Performance benchmark
    print("\n[3] Performance Benchmark")
    
    for t0 in [1e4, 1e5, 1e6, 1e7]:
        # Parameters
        density = math.log(t0) / TWO_PI
        num_t = 10000
        delta = 1.0 / (10 * density)  # 10 points per zero
        M = int(math.sqrt(t0 / TWO_PI)) + 10
        
        # Warmup
        engine.evaluate(t0, delta, 100, min(M, 1000))
        torch.cuda.synchronize()
        
        # Time it
        start = time.time()
        t_vals, Z_vals = engine.evaluate_Z(t0, delta, num_t)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Count zeros
        signs = torch.sign(Z_vals)
        zeros = ((signs[:-1] * signs[1:]) < 0).sum().item()
        
        evals_per_sec = num_t / elapsed
        zeros_per_sec = zeros / elapsed if elapsed > 0 else 0
        
        print(f"  t={t0:.0e}: M={M:>6}, {zeros:>4} zeros, "
              f"{elapsed:.3f}s, {evals_per_sec:,.0f} evals/s")
    
    # Scaling comparison
    print("\n[4] Scaling: Direct vs Range Decomposition")
    
    t0 = 1e5
    num_t = 1000
    delta = 0.1
    M = int(math.sqrt(t0 / TWO_PI)) + 10
    
    # Direct O(N*M)
    torch.cuda.synchronize()
    start = time.time()
    
    n = torch.arange(1, M + 1, device=device, dtype=torch.float64)
    ln_n = torch.log(n)
    rsqrt_n = torch.rsqrt(n)
    
    S_direct = torch.zeros(num_t, device=device, dtype=torch.complex128)
    for ki in range(num_t):
        t = t0 + ki * delta
        phases = t * ln_n
        S_direct[ki] = (rsqrt_n * torch.exp(1j * phases)).sum()
    
    torch.cuda.synchronize()
    direct_time = time.time() - start
    
    # Range decomposition O(M log M)
    torch.cuda.synchronize()
    start = time.time()
    S_range = engine.evaluate(t0, delta, num_t, M)
    torch.cuda.synchronize()
    range_time = time.time() - start
    
    print(f"  Direct:  {direct_time:.3f}s")
    print(f"  Range:   {range_time:.3f}s")
    print(f"  Speedup: {direct_time/range_time:.1f}x")
    
    # Projections
    print("\n" + "="*70)
    print("PROJECTIONS")
    print("="*70)
    
    # Use measured performance to project
    # At t=10^7, we got some evals/sec
    # Scale by M log M / M' log M' for different t
    
    base_rate = evals_per_sec  # from t=10^7 benchmark
    
    print(f"\nBased on measured {base_rate:,.0f} evals/sec at t=10^7:")
    
    for target_zeros, name in [(1e9, "10^9"), (1e12, "10^12"), (1e13, "10^13"), (1e16, "10^16")]:
        # t needed for target zeros
        t_target = target_zeros * TWO_PI / math.log(target_zeros) * 2  # rough
        M_target = math.sqrt(t_target / TWO_PI)
        
        # Scaling factor (range decomposition scales as M log M)
        M_base = math.sqrt(1e7 / TWO_PI)
        scale = (M_target * math.log(M_target + 1)) / (M_base * math.log(M_base + 1))
        
        adjusted_rate = base_rate / scale
        
        # Need 10 * target_zeros evaluations
        total_evals = 10 * target_zeros
        time_sec = total_evals / adjusted_rate
        
        if time_sec < 60:
            time_str = f"{time_sec:.0f} sec"
        elif time_sec < 3600:
            time_str = f"{time_sec/60:.0f} min"
        elif time_sec < 86400:
            time_str = f"{time_sec/3600:.1f} hours"
        else:
            time_str = f"{time_sec/86400:.1f} days"
        
        print(f"  {name}: M={M_target:.0e}, ~{time_str}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    benchmark()
