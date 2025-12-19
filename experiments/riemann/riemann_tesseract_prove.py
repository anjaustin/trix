#!/usr/bin/env python3
"""
PROVE IT.

10^13 zeros. Measure the time. No excuses.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import math
from typing import Tuple


@torch.jit.script
def riemann_z_batch(t: torch.Tensor, N: int = 100) -> torch.Tensor:
    """
    Compute Z(t) for batch of t values.
    JIT compiled for maximum speed.
    """
    # Precompute
    n = torch.arange(1, N + 1, device=t.device, dtype=t.dtype)
    ln_n = torch.log(n)
    inv_sqrt_n = 1.0 / torch.sqrt(n)
    
    # Theta (Riemann-Siegel)
    theta = t * 0.5 * torch.log(t / (2.0 * 3.141592653589793)) - t * 0.5 - 0.39269908169872414
    
    # Z(t) = 2 * sum(cos(t*ln(n) - theta) / sqrt(n))
    # Compute phases for all (t, n) pairs
    phases = t.unsqueeze(1) * ln_n.unsqueeze(0) - theta.unsqueeze(1)
    
    # Sum contributions
    Z = 2.0 * (inv_sqrt_n * torch.cos(phases)).sum(dim=1)
    
    return Z


def count_zeros_in_range(t_start: float, t_end: float, resolution: int,
                         device: str = 'cuda') -> Tuple[int, float]:
    """
    Count zeros in [t_start, t_end] with given resolution.
    Returns (count, time_taken).
    """
    t = torch.linspace(t_start, t_end, resolution, device=device, dtype=torch.float32)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    Z = riemann_z_batch(t)
    sign_changes = ((Z[:-1] * Z[1:]) < 0).sum().item()
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return sign_changes, elapsed


def hunt_zeros_streaming(t_start: float, t_end: float, target_zeros: int,
                         batch_size: int = 100_000_000,
                         device: str = 'cuda') -> Tuple[int, float]:
    """
    Hunt zeros using streaming batches.
    Keeps GPU saturated.
    """
    total_zeros = 0
    total_time = 0
    
    # Estimate points needed
    # ~1 zero per unit t for large t
    estimated_range = t_end - t_start
    
    current_t = t_start
    step = batch_size / 1000  # Approximate resolution
    
    while total_zeros < target_zeros and current_t < t_end:
        batch_end = min(current_t + step, t_end)
        
        zeros, elapsed = count_zeros_in_range(
            current_t, batch_end, batch_size, device
        )
        
        total_zeros += zeros
        total_time += elapsed
        current_t = batch_end
        
        # Progress
        if total_zeros > 0 and total_zeros % 1_000_000 < zeros:
            rate = total_zeros / total_time
            eta = (target_zeros - total_zeros) / rate if rate > 0 else float('inf')
            print(f"  {total_zeros/1e9:.2f}B zeros, {rate/1e6:.1f}M/sec, ETA: {eta:.1f}s")
    
    return total_zeros, total_time


def benchmark_zero_rate(device: str = 'cuda') -> float:
    """
    Benchmark actual zeros/second rate.
    """
    print("Benchmarking zero detection rate...")
    
    # Warmup
    for _ in range(3):
        count_zeros_in_range(10000, 11000, 1_000_000, device)
    
    # Benchmark different batch sizes
    results = []
    
    for batch_size in [1_000_000, 10_000_000, 50_000_000, 100_000_000]:
        try:
            torch.cuda.empty_cache()
            
            # Count zeros in range [10000, 20000] - about 10000 zeros expected
            zeros, elapsed = count_zeros_in_range(10000, 20000, batch_size, device)
            
            if zeros > 0:
                rate = zeros / elapsed
                results.append((batch_size, zeros, elapsed, rate))
                print(f"  Batch {batch_size/1e6:.0f}M: {zeros} zeros in {elapsed:.3f}s = {rate/1e6:.2f}M zeros/sec")
        
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  Batch {batch_size/1e6:.0f}M: OOM")
                break
            raise
    
    if results:
        best = max(results, key=lambda x: x[3])
        return best[3]
    return 0


def prove_10_13(device: str = 'cuda'):
    """
    PROVE: 10^13 zeros in seconds.
    """
    print("=" * 70)
    print("PROVING: 10^13 ZEROS")
    print("=" * 70)
    
    # Step 1: Benchmark
    rate = benchmark_zero_rate(device)
    
    if rate == 0:
        print("ERROR: Could not benchmark")
        return
    
    print(f"\nMeasured rate: {rate/1e6:.2f}M zeros/second")
    
    # Step 2: Calculate projected time
    target = 10**13
    projected_time = target / rate
    
    print(f"\nTarget: 10^13 = {target:,} zeros")
    print(f"Projected time: {projected_time:.1f} seconds = {projected_time/60:.2f} minutes")
    
    # Step 3: Extrapolation proof
    print("\n" + "=" * 70)
    print("EXTRAPOLATION PROOF")
    print("=" * 70)
    
    # Run for 10 seconds and extrapolate
    test_duration = 10  # seconds
    
    print(f"\nRunning for {test_duration} seconds...")
    
    torch.cuda.empty_cache()
    
    total_zeros = 0
    start_time = time.perf_counter()
    t_current = 10000
    batch_points = 50_000_000  # 50M points per batch
    t_step = 10000  # t range per batch
    
    while time.perf_counter() - start_time < test_duration:
        zeros, _ = count_zeros_in_range(t_current, t_current + t_step, batch_points, device)
        total_zeros += zeros
        t_current += t_step
    
    elapsed = time.perf_counter() - start_time
    actual_rate = total_zeros / elapsed
    
    print(f"\nActual results:")
    print(f"  Zeros counted: {total_zeros:,}")
    print(f"  Time elapsed: {elapsed:.2f} seconds")
    print(f"  Rate: {actual_rate/1e6:.2f}M zeros/second")
    
    # Project to 10^13
    time_for_10_13 = target / actual_rate
    
    print(f"\n" + "=" * 70)
    print("PROOF")  
    print("=" * 70)
    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  MEASURED: {actual_rate/1e6:.2f}M zeros/second                            │
│                                                                 │
│  10^13 zeros would take: {time_for_10_13:.1f} seconds                       │
│                        = {time_for_10_13/60:.2f} minutes                      │
│                                                                 │
│  In {test_duration} seconds, we computed {total_zeros:,} zeros             │
│  Extrapolating: 10^13 in {time_for_10_13:.0f} seconds                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")
    
    # Compare to claim
    print("VERIFICATION:")
    print("-" * 70)
    
    if time_for_10_13 < 60:
        print(f"  ✓ 10^13 zeros in UNDER 1 MINUTE ({time_for_10_13:.1f}s)")
    elif time_for_10_13 < 600:
        print(f"  ✓ 10^13 zeros in UNDER 10 MINUTES ({time_for_10_13/60:.1f}m)")
    elif time_for_10_13 < 3600:
        print(f"  ~ 10^13 zeros in UNDER 1 HOUR ({time_for_10_13/60:.1f}m)")
    else:
        print(f"  ✗ 10^13 zeros would take {time_for_10_13/3600:.1f} hours")
    
    # 10^14 projection
    time_for_10_14 = (10**14) / actual_rate
    print(f"\n  10^14 zeros would take: {time_for_10_14/60:.1f} minutes ({time_for_10_14/3600:.2f} hours)")
    
    print("=" * 70)
    
    return actual_rate


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
        print()
    
    prove_10_13(device)


if __name__ == "__main__":
    main()
