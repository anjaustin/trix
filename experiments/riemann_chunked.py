#!/usr/bin/env python3
"""
RIEMANN CHUNKED - Memory efficient with max throughput

The trick: chunk over N (small), stream over B (large)
"""

import torch
import time
import math


def count_zeros_chunked(t_start: float, t_end: float, num_points: int,
                        N: int = 50, chunk_n: int = 50,
                        device: str = 'cuda', dtype=torch.float32) -> tuple:
    """
    Memory-efficient Z(t) computation.
    
    Instead of [B, N] matrix, compute in chunks and accumulate.
    """
    B = num_points
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # t values [B]
    t = torch.linspace(t_start, t_end, B, device=device, dtype=dtype)
    
    # Theta [B] - compute once
    PI = math.pi
    if dtype == torch.float16:
        t_f32 = t.float()
        theta = (0.5 * t_f32 * (torch.log(t_f32) - 1.8378770664093453) - 0.5 * t_f32 - PI / 8.0).to(dtype)
    else:
        theta = 0.5 * t * (torch.log(t) - 1.8378770664093453) - 0.5 * t - PI / 8.0
    
    # Accumulator [B]
    Z = torch.zeros(B, device=device, dtype=dtype)
    
    # Process n in chunks
    for n_start in range(0, N, chunk_n):
        n_end = min(n_start + chunk_n, N)
        chunk_size = n_end - n_start
        
        # n values for this chunk [chunk_size]
        n = torch.arange(n_start + 1, n_end + 1, device=device, dtype=dtype)
        ln_n = torch.log(n)  # [chunk_size]
        inv_sqrt_n = torch.rsqrt(n)  # [chunk_size]
        
        # Phase matrix [B, chunk_size]
        phases = t.unsqueeze(1) * ln_n.unsqueeze(0) - theta.unsqueeze(1)
        
        # Cosine and weighted sum
        cos_phases = torch.cos(phases)
        Z += (cos_phases * inv_sqrt_n.unsqueeze(0)).sum(dim=1)
    
    Z = 2.0 * Z
    
    # Sign changes
    signs = torch.sign(Z.float() if dtype == torch.float16 else Z)
    changes = ((signs[:-1] * signs[1:]) < 0).sum().item()
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return changes, elapsed


def benchmark():
    device = 'cuda'
    
    print("=" * 70)
    print("RIEMANN CHUNKED - MEMORY EFFICIENT")
    print("=" * 70)
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory/1e9:.1f} GB")
    
    # Warmup
    print("\nWarmup...")
    for _ in range(5):
        count_zeros_chunked(10000, 20000, 1000000, N=50, device=device)
    
    # Benchmark FP32
    print("\n--- FP32 ---")
    best_fp32 = 0
    for num_points in [1_000_000, 10_000_000, 50_000_000, 100_000_000, 200_000_000]:
        try:
            torch.cuda.empty_cache()
            zeros, elapsed = count_zeros_chunked(10000, 100000, num_points, N=50, device=device, dtype=torch.float32)
            rate = zeros / elapsed
            print(f"  {num_points/1e6:6.0f}M: {zeros:6d} zeros in {elapsed:.3f}s = {rate/1e6:8.2f}M/sec")
            best_fp32 = max(best_fp32, rate)
        except RuntimeError:
            print(f"  {num_points/1e6:6.0f}M: OOM")
            break
    
    # Benchmark FP16
    print("\n--- FP16 ---")
    best_fp16 = 0
    for num_points in [1_000_000, 10_000_000, 50_000_000, 100_000_000, 200_000_000, 500_000_000]:
        try:
            torch.cuda.empty_cache()
            zeros, elapsed = count_zeros_chunked(10000, 100000, num_points, N=50, device=device, dtype=torch.float16)
            rate = zeros / elapsed
            print(f"  {num_points/1e6:6.0f}M: {zeros:6d} zeros in {elapsed:.3f}s = {rate/1e6:8.2f}M/sec")
            best_fp16 = max(best_fp16, rate)
        except RuntimeError:
            print(f"  {num_points/1e6:6.0f}M: OOM")
            break
    
    # Find optimal batch size for FP16
    print("\n--- Finding optimal batch size (FP16) ---")
    optimal_batch = 10_000_000
    optimal_rate = 0
    
    for num_points in [5_000_000, 10_000_000, 20_000_000, 50_000_000]:
        try:
            torch.cuda.empty_cache()
            zeros, elapsed = count_zeros_chunked(10000, 100000, num_points, N=50, device=device, dtype=torch.float16)
            rate = zeros / elapsed
            if rate > optimal_rate:
                optimal_rate = rate
                optimal_batch = num_points
        except:
            break
    
    print(f"  Optimal: {optimal_batch/1e6:.0f}M points @ {optimal_rate/1e6:.2f}M zeros/sec")
    
    # Extended run with optimal batch
    print("\n" + "=" * 70)
    print(f"EXTENDED RUN (60 seconds) - FP16, {optimal_batch/1e6:.0f}M batch")
    print("=" * 70)
    
    total_zeros = 0
    total_batches = 0
    start_total = time.perf_counter()
    t_current = 10000
    t_step = 100000
    
    while time.perf_counter() - start_total < 60:
        try:
            zeros, _ = count_zeros_chunked(t_current, t_current + t_step, optimal_batch, 
                                           N=50, device=device, dtype=torch.float16)
            total_zeros += zeros
            total_batches += 1
            t_current += t_step
            
            if total_batches % 100 == 0:
                elapsed = time.perf_counter() - start_total
                rate = total_zeros / elapsed
                print(f"  Batch {total_batches}: {total_zeros:,} zeros, {rate/1e6:.2f}M/sec")
        except:
            break
    
    elapsed_total = time.perf_counter() - start_total
    final_rate = total_zeros / elapsed_total
    
    print(f"\nFinal: {total_zeros:,} zeros in {elapsed_total:.1f}s = {final_rate/1e6:.2f}M zeros/sec")
    
    # Projections
    print("\n" + "=" * 70)
    print("PROJECTIONS")
    print("=" * 70)
    
    for target, name in [(10**9, "10^9"), (10**10, "10^10"), (10**11, "10^11"), 
                         (10**12, "10^12"), (10**13, "10^13")]:
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
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    baseline = 0.45e6
    print(f"Previous best:  {baseline/1e6:.2f}M zeros/sec")
    print(f"FP16 chunked:   {final_rate/1e6:.2f}M zeros/sec")
    print(f"Speedup:        {final_rate/baseline:.1f}x")
    
    return final_rate


if __name__ == "__main__":
    benchmark()
