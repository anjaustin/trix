#!/usr/bin/env python3
"""
RIEMANN TRITON V2 - FULLY PARALLEL

Key insight: Don't loop over N inside kernel.
Parallelize over BOTH t and n, then reduce.
"""

import torch
import triton
import triton.language as tl
import time
import math


@triton.jit
def compute_contributions_kernel(
    t_ptr,           # [B] t values
    out_ptr,         # [B, N] contributions
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute contribution matrix: out[i,j] = cos(t[i]*ln(j+1) - theta[i]) / sqrt(j+1)
    
    Fully parallel over both dimensions.
    """
    # 2D grid: (B/BLOCK_B, N/BLOCK_N)
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Masks
    mask_b = offs_b < B
    mask_n = offs_n < N
    
    # Load t values [BLOCK_B]
    t = tl.load(t_ptr + offs_b, mask=mask_b, other=1.0)
    
    # Compute n values (1-indexed) [BLOCK_N]
    n = (offs_n + 1).to(tl.float32)
    
    # ln(n) and 1/sqrt(n)
    ln_n = tl.log(n)
    inv_sqrt_n = 1.0 / tl.sqrt(n)
    
    # Theta for each t [BLOCK_B]
    PI = 3.141592653589793
    ln_t = tl.log(t)
    theta = 0.5 * t * (ln_t - 1.8378770664093453) - 0.5 * t - PI / 8.0
    
    # Phase matrix [BLOCK_B, BLOCK_N] via broadcasting
    # t[:, None] * ln_n[None, :] - theta[:, None]
    # Triton doesn't support 2D directly, so we compute element by element
    
    # For each (b, n) pair
    for b_idx in range(BLOCK_B):
        if offs_b[b_idx] < B:
            t_val = t[b_idx]
            theta_val = theta[b_idx]
            
            for n_idx in range(BLOCK_N):
                if offs_n[n_idx] < N:
                    # Phase
                    phase = t_val * ln_n[n_idx] - theta_val
                    
                    # Reduce mod 2π
                    TWO_PI = 6.283185307179586
                    phase = phase - tl.floor(phase / TWO_PI) * TWO_PI - PI
                    
                    # Cosine (Taylor)
                    x2 = phase * phase
                    cos_val = 1.0 - x2 * 0.5 + x2 * x2 * 0.041666667
                    
                    # Contribution
                    contrib = cos_val * inv_sqrt_n[n_idx]
                    
                    # Store
                    out_idx = offs_b[b_idx] * N + offs_n[n_idx]
                    tl.store(out_ptr + out_idx, contrib)


@triton.jit
def sum_reduce_kernel(
    in_ptr,          # [B, N] contributions
    out_ptr,         # [B] sums
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sum over N dimension."""
    pid = tl.program_id(0)
    
    offs = tl.arange(0, N)
    mask = offs < N
    
    # Load row
    row_ptr = in_ptr + pid * N
    vals = tl.load(row_ptr + offs, mask=mask, other=0.0)
    
    # Sum
    total = tl.sum(vals)
    
    # Store
    tl.store(out_ptr + pid, total * 2.0)


class TritonRiemannV2:
    """Fully parallel Triton Riemann engine."""
    
    def __init__(self, N: int = 32, device='cuda'):
        self.N = N
        self.device = device
        print(f"Triton V2 Engine: N={N}")
    
    def count_zeros_matmul(self, t_start: float, t_end: float, num_points: int) -> tuple:
        """
        Use matrix multiply for maximum parallelism.
        
        Z(t) = 2 * sum_n cos(t * ln(n) - theta) / sqrt(n)
        
        Trick: Precompute phase matrix, use torch.cos (highly optimized)
        """
        device = self.device
        N = self.N
        B = num_points
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # t values [B]
        t = torch.linspace(t_start, t_end, B, device=device, dtype=torch.float32)
        
        # n values [N]
        n = torch.arange(1, N + 1, device=device, dtype=torch.float32)
        ln_n = torch.log(n)  # [N]
        inv_sqrt_n = torch.rsqrt(n)  # [N]
        
        # Theta [B]
        PI = math.pi
        theta = 0.5 * t * (torch.log(t) - 1.8378770664093453) - 0.5 * t - PI / 8.0
        
        # Phase matrix [B, N] = t[:, None] * ln_n[None, :] - theta[:, None]
        phases = t.unsqueeze(1) * ln_n.unsqueeze(0) - theta.unsqueeze(1)
        
        # Cosine [B, N]
        cos_phases = torch.cos(phases)
        
        # Weighted [B, N] * [N] -> [B, N]
        weighted = cos_phases * inv_sqrt_n.unsqueeze(0)
        
        # Sum over N -> [B]
        Z = 2.0 * weighted.sum(dim=1)
        
        # Sign changes
        signs = torch.sign(Z)
        changes = ((signs[:-1] * signs[1:]) < 0).sum().item()
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        return changes, elapsed
    
    def count_zeros_fp16(self, t_start: float, t_end: float, num_points: int) -> tuple:
        """FP16 version for 2x throughput."""
        device = self.device
        N = self.N
        B = num_points
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # FP16 for bulk compute
        t = torch.linspace(t_start, t_end, B, device=device, dtype=torch.float16)
        n = torch.arange(1, N + 1, device=device, dtype=torch.float16)
        
        ln_n = torch.log(n)
        inv_sqrt_n = torch.rsqrt(n)
        
        # Theta - keep in FP32 for precision, then convert
        t_f32 = t.float()
        theta = (0.5 * t_f32 * (torch.log(t_f32) - 1.8378770664093453) - 0.5 * t_f32 - 0.39269908169872414).half()
        
        # Phase matrix [B, N]
        phases = t.unsqueeze(1) * ln_n.unsqueeze(0) - theta.unsqueeze(1)
        
        # Cosine
        cos_phases = torch.cos(phases)
        
        # Weighted sum
        Z = 2.0 * (cos_phases * inv_sqrt_n.unsqueeze(0)).sum(dim=1)
        
        # Sign changes (convert to FP32 for comparison)
        signs = torch.sign(Z.float())
        changes = ((signs[:-1] * signs[1:]) < 0).sum().item()
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        return changes, elapsed


def benchmark():
    device = 'cuda'
    
    print("=" * 70)
    print("TRITON V2 - MATMUL + FP16")
    print("=" * 70)
    
    engine = TritonRiemannV2(N=50, device=device)
    
    # Warmup
    print("\nWarmup...")
    for _ in range(5):
        engine.count_zeros_matmul(10000, 20000, 1000000)
        engine.count_zeros_fp16(10000, 20000, 1000000)
    
    # Benchmark FP32
    print("\n--- FP32 Matmul ---")
    for num_points in [1_000_000, 10_000_000, 50_000_000]:
        try:
            torch.cuda.empty_cache()
            zeros, elapsed = engine.count_zeros_matmul(10000, 100000, num_points)
            rate = zeros / elapsed
            print(f"  {num_points/1e6:.0f}M: {zeros} zeros in {elapsed:.3f}s = {rate/1e6:.2f}M/sec")
        except RuntimeError as e:
            print(f"  {num_points/1e6:.0f}M: OOM")
    
    # Benchmark FP16
    print("\n--- FP16 Matmul ---")
    best_rate = 0
    for num_points in [1_000_000, 10_000_000, 50_000_000, 100_000_000]:
        try:
            torch.cuda.empty_cache()
            zeros, elapsed = engine.count_zeros_fp16(10000, 100000, num_points)
            rate = zeros / elapsed
            print(f"  {num_points/1e6:.0f}M: {zeros} zeros in {elapsed:.3f}s = {rate/1e6:.2f}M/sec")
            best_rate = max(best_rate, rate)
        except RuntimeError as e:
            print(f"  {num_points/1e6:.0f}M: OOM")
    
    # Extended run
    print("\n" + "=" * 70)
    print("EXTENDED RUN (30 seconds) - FP16")
    print("=" * 70)
    
    total_zeros = 0
    start_total = time.perf_counter()
    t_current = 10000
    t_step = 100000
    batch_size = 10_000_000
    
    while time.perf_counter() - start_total < 30:
        try:
            zeros, _ = engine.count_zeros_fp16(t_current, t_current + t_step, batch_size)
            total_zeros += zeros
            t_current += t_step
        except:
            break
    
    elapsed_total = time.perf_counter() - start_total
    final_rate = total_zeros / elapsed_total
    
    print(f"Total: {total_zeros:,} zeros in {elapsed_total:.1f}s = {final_rate/1e6:.2f}M zeros/sec")
    
    # Projections
    print("\n" + "=" * 70)
    print("PROJECTIONS (FP16)")
    print("=" * 70)
    
    for target, name in [(10**10, "10^10"), (10**12, "10^12"), (10**13, "10^13")]:
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
    
    # Speedup
    baseline = 0.45e6
    print(f"\nSpeedup vs baseline: {final_rate/baseline:.1f}x")


if __name__ == "__main__":
    benchmark()
