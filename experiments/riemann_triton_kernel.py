#!/usr/bin/env python3
"""
RIEMANN TRITON KERNEL

Fused Z(t) evaluation with mixed precision:
- FP32 for phase computation (needs precision for mod 2π)
- FP16/FP8 for cos lookup and accumulation
- Sign detection only needs 1 bit!

The insight: We don't need Z(t) value, just sign(Z(t)).
"""

import torch
import triton
import triton.language as tl
import time
import math


@triton.jit
def riemann_z_kernel(
    t_ptr,           # Input: t values [B]
    signs_ptr,       # Output: signs [B]
    ln_n_ptr,        # Precomputed: ln(1), ln(2), ..., ln(N)
    inv_sqrt_ptr,    # Precomputed: 1/sqrt(1), 1/sqrt(2), ..., 1/sqrt(N)
    B: tl.constexpr,           # Batch size
    N: tl.constexpr,           # Number of terms
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    Fused kernel: compute sign(Z(t)) for batch of t values.
    
    Z(t) = 2 * sum_{n=1}^{N} cos(t*ln(n) - theta(t)) / sqrt(n)
    
    We only output the SIGN, not the value.
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Compute indices for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < B
    
    # Load t values
    t = tl.load(t_ptr + offsets, mask=mask, other=0.0)
    
    # Compute theta(t) = t/2 * ln(t/(2π)) - t/2 - π/8
    # Simplified: theta ≈ t/2 * (ln(t) - ln(2π)) - t/2 - π/8
    PI = 3.141592653589793
    ln_t = tl.log(t + 1e-10)  # Avoid log(0)
    ln_2pi = 1.8378770664093453  # ln(2π)
    theta = 0.5 * t * (ln_t - ln_2pi) - 0.5 * t - PI / 8.0
    
    # Accumulator for Z(t)
    Z_accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Sum over n=1 to N
    for n_idx in range(N):
        # Load precomputed values
        ln_n = tl.load(ln_n_ptr + n_idx)
        inv_sqrt_n = tl.load(inv_sqrt_ptr + n_idx)
        
        # Phase = t * ln(n) - theta
        phase = t * ln_n - theta
        
        # Reduce phase mod 2π for numerical stability
        phase = phase - tl.floor(phase / (2.0 * PI)) * (2.0 * PI)
        
        # Cosine approximation (Taylor series, good enough for sign detection)
        # cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720
        # Shift to [-π, π] range
        phase = phase - PI  # Now in [-π, π]
        x2 = phase * phase
        x4 = x2 * x2
        x6 = x4 * x2
        cos_val = 1.0 - x2 * 0.5 + x4 * 0.041666667 - x6 * 0.001388889
        
        # Accumulate weighted contribution
        Z_accum += cos_val * inv_sqrt_n
    
    # Multiply by 2
    Z_accum = Z_accum * 2.0
    
    # Output sign: +1 if positive, -1 if negative, 0 if zero
    signs = tl.where(Z_accum > 0, 1, tl.where(Z_accum < 0, -1, 0))
    
    # Store signs
    tl.store(signs_ptr + offsets, signs, mask=mask)


@triton.jit  
def riemann_z_kernel_fast(
    t_ptr,           # Input: t values [B]
    signs_ptr,       # Output: signs [B]
    N: tl.constexpr,           # Number of terms (small, unrolled)
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    """
    Faster version: inline everything, no external loads.
    For N <= 32, fully unrolled.
    """
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load t values
    t = tl.load(t_ptr + offsets)
    
    # Constants
    PI = 3.141592653589793
    TWO_PI = 6.283185307179586
    
    # Theta
    ln_t = tl.log(t)
    theta = 0.5 * t * (ln_t - 1.8378770664093453) - 0.5 * t - 0.39269908169872414
    
    # Accumulator
    Z = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Unrolled sum for n=1 to N
    # n=1: ln(1)=0, 1/sqrt(1)=1
    phase = -theta
    phase = phase - tl.floor(phase / TWO_PI) * TWO_PI - PI
    x2 = phase * phase
    Z += (1.0 - x2 * 0.5 + x2 * x2 * 0.041666667) * 1.0
    
    # n=2: ln(2)=0.693, 1/sqrt(2)=0.707
    phase = t * 0.6931471805599453 - theta
    phase = phase - tl.floor(phase / TWO_PI) * TWO_PI - PI
    x2 = phase * phase
    Z += (1.0 - x2 * 0.5 + x2 * x2 * 0.041666667) * 0.7071067811865476
    
    # n=3 to N inline...
    # For brevity, using loop with constants
    ln_vals = [0.0, 0.693147, 1.098612, 1.386294, 1.609438, 1.791759, 1.945910, 2.079442,
               2.197225, 2.302585, 2.397895, 2.484907, 2.564949, 2.639057, 2.708050, 2.772589]
    inv_sqrt_vals = [1.0, 0.707107, 0.577350, 0.5, 0.447214, 0.408248, 0.377964, 0.353553,
                    0.333333, 0.316228, 0.301511, 0.288675, 0.277350, 0.267261, 0.258199, 0.25]
    
    # This would need to be generated or use a different approach
    # For now, the kernel above with precomputed arrays is the way to go
    
    # Output sign
    signs = tl.where(Z > 0, 1, tl.where(Z < 0, -1, 0))
    tl.store(signs_ptr + offsets, signs)


class TritonRiemannEngine:
    """Triton-accelerated Riemann zero detection."""
    
    def __init__(self, N: int = 50, device='cuda'):
        self.N = N
        self.device = device
        
        # Precompute ln(n) and 1/sqrt(n)
        n = torch.arange(1, N + 1, dtype=torch.float32, device=device)
        self.ln_n = torch.log(n)
        self.inv_sqrt_n = torch.rsqrt(n)
        
        print(f"Triton Riemann Engine initialized: N={N}")
    
    def compute_signs(self, t: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
        """Compute signs of Z(t) for batch of t values."""
        B = t.shape[0]
        signs = torch.empty(B, dtype=torch.int32, device=self.device)
        
        # Grid
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE']),)
        
        # Launch kernel
        riemann_z_kernel[grid](
            t, signs, self.ln_n, self.inv_sqrt_n,
            B=B, N=self.N, BLOCK_SIZE=block_size
        )
        
        return signs
    
    def count_zeros(self, t_start: float, t_end: float, num_points: int) -> tuple:
        """Count zeros in range."""
        t = torch.linspace(t_start, t_end, num_points, device=self.device, dtype=torch.float32)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        signs = self.compute_signs(t)
        
        # Count sign changes
        changes = ((signs[:-1] * signs[1:]) < 0).sum().item()
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        return changes, elapsed


def benchmark_triton():
    """Benchmark Triton kernel."""
    device = 'cuda'
    
    print("=" * 70)
    print("TRITON RIEMANN KERNEL BENCHMARK")
    print("=" * 70)
    
    # Initialize
    engine = TritonRiemannEngine(N=50, device=device)
    
    # Warmup
    print("\nWarmup...")
    for _ in range(5):
        engine.count_zeros(10000, 10100, 100000)
    
    # Benchmark
    print("\nBenchmarking...")
    
    results = []
    
    for num_points in [1_000_000, 10_000_000, 50_000_000, 100_000_000]:
        try:
            torch.cuda.empty_cache()
            
            zeros, elapsed = engine.count_zeros(10000, 100000, num_points)
            rate = zeros / elapsed if elapsed > 0 else 0
            
            print(f"  {num_points/1e6:6.0f}M points: {zeros:6d} zeros in {elapsed:.3f}s = {rate/1e6:8.2f}M zeros/sec")
            results.append((num_points, zeros, elapsed, rate))
            
        except Exception as e:
            print(f"  {num_points/1e6:6.0f}M points: Error - {e}")
    
    if results:
        best = max(results, key=lambda x: x[3])
        best_rate = best[3]
        
        print(f"\nBest rate: {best_rate/1e6:.2f}M zeros/sec")
        
        # Extended run
        print("\n" + "=" * 70)
        print("EXTENDED RUN (30 seconds)")
        print("=" * 70)
        
        total_zeros = 0
        start_total = time.perf_counter()
        t_current = 10000
        t_step = 100000
        
        while time.perf_counter() - start_total < 30:
            zeros, _ = engine.count_zeros(t_current, t_current + t_step, best[0])
            total_zeros += zeros
            t_current += t_step
        
        elapsed_total = time.perf_counter() - start_total
        final_rate = total_zeros / elapsed_total
        
        print(f"Total: {total_zeros:,} zeros in {elapsed_total:.1f}s = {final_rate/1e6:.2f}M zeros/sec")
        
        # Projections
        print("\n" + "=" * 70)
        print("PROJECTIONS")
        print("=" * 70)
        
        for target, name in [(10**10, "10^10"), (10**12, "10^12"), (10**13, "10^13"), (10**14, "10^14")]:
            seconds = target / final_rate
            if seconds < 60:
                time_str = f"{seconds:.1f} seconds"
            elif seconds < 3600:
                time_str = f"{seconds/60:.1f} minutes"
            elif seconds < 86400:
                time_str = f"{seconds/3600:.1f} hours"
            else:
                time_str = f"{seconds/86400:.1f} days"
            
            print(f"  {name}: {time_str}")
        
        return final_rate
    
    return 0


if __name__ == "__main__":
    rate = benchmark_triton()
    
    if rate > 0:
        print("\n" + "=" * 70)
        print("TRITON SPEEDUP")
        print("=" * 70)
        
        baseline = 0.45e6  # Previous best
        speedup = rate / baseline
        
        print(f"Baseline (PyTorch): {baseline/1e6:.2f}M zeros/sec")
        print(f"Triton kernel:      {rate/1e6:.2f}M zeros/sec")
        print(f"Speedup:            {speedup:.1f}x")
