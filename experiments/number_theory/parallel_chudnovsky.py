#!/usr/bin/env python3
"""
Mesa 10 Parallel: Multi-core Binary Splitting Chudnovsky

Parallelizes the binary splitting algorithm across CPU cores using
Python multiprocessing + GMP (via gmpy2).

Key insight: In the binary splitting tree, left and right children
can be computed in parallel. With depth-first parallelization, we
achieve near-linear speedup with core count.

Architecture:
                        [Root]
                       /      \
                    [L]        [R]       ← PARALLEL
                   /   \      /   \
                [LL] [LR]  [RL] [RR]     ← PARALLEL
                
Each subtree computes (P, Q, T) independently.
Merge happens sequentially at each level.

Performance targets:
- 1 core:  1.75M digits/sec (baseline)
- 8 cores: ~10M digits/sec (5.7x speedup)
- Amdahl's law limits: ~80% parallelizable
"""

import os
import time
import multiprocessing as mp
from typing import Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import gmpy2
from gmpy2 import mpz, mpfr

# Chudnovsky constants
A = mpz(13591409)
B = mpz(545140134)
C = mpz(640320)
C3 = C ** 3
C3_24 = C3 // 24


def _binary_split(a: int, b: int) -> Tuple[mpz, mpz, mpz]:
    """
    Single-threaded binary splitting for range [a, b).
    
    Returns (P, Q, T) where:
    - P: Product term
    - Q: Denominator term  
    - T: Sum term (numerator)
    """
    if b - a == 1:
        if a == 0:
            return mpz(1), mpz(1), A
        else:
            k = a
            Pab = mpz(6*k - 5) * mpz(2*k - 1) * mpz(6*k - 1)
            Qab = mpz(k) ** 3 * C3_24
            Tab = Pab * (A + B * mpz(k))
            if k & 1:
                Tab = -Tab
            return Pab, Qab, Tab
    
    m = (a + b) // 2
    Pam, Qam, Tam = _binary_split(a, m)
    Pmb, Qmb, Tmb = _binary_split(m, b)
    
    return Pam * Pmb, Qam * Qmb, Qmb * Tam + Pam * Tmb


def _worker_binary_split(args: Tuple[int, int]) -> Tuple[bytes, bytes, bytes]:
    """
    Worker function for parallel execution.
    
    Converts results to bytes for pickling (mpz isn't directly picklable).
    """
    a, b = args
    P, Q, T = _binary_split(a, b)
    # Convert to bytes for pickling
    return (gmpy2.to_binary(P), gmpy2.to_binary(Q), gmpy2.to_binary(T))


def _merge_results(left: Tuple[mpz, mpz, mpz], right: Tuple[mpz, mpz, mpz]) -> Tuple[mpz, mpz, mpz]:
    """Merge two binary splitting results."""
    Pam, Qam, Tam = left
    Pmb, Qmb, Tmb = right
    return Pam * Pmb, Qam * Qmb, Qmb * Tam + Pam * Tmb


class ParallelChudnovsky:
    """
    Parallel binary splitting Chudnovsky algorithm.
    
    Uses multiprocessing to compute independent subtrees in parallel.
    """
    
    def __init__(self, precision: int, num_workers: int = None):
        """
        Initialize parallel Chudnovsky.
        
        Args:
            precision: Number of decimal digits to compute
            num_workers: Number of parallel workers (default: CPU count)
        """
        self.precision = precision
        self.num_terms = precision // 14 + 10
        self.num_workers = num_workers or mp.cpu_count()
        self.bits = int(precision * 4) + 1000
    
    def compute(self) -> str:
        """
        Compute π using parallel binary splitting.
        
        Returns:
            String of π decimal digits (after "3.")
        """
        gmpy2.get_context().precision = self.bits
        
        n = self.num_terms
        
        # Determine parallelization depth
        # We want enough chunks for all workers, but not too many (overhead)
        num_chunks = min(self.num_workers * 4, n)
        chunk_size = max(1, n // num_chunks)
        
        # Create chunk boundaries
        chunks = []
        for i in range(0, n, chunk_size):
            chunks.append((i, min(i + chunk_size, n)))
        
        # If too few chunks, fall back to sequential
        if len(chunks) <= 1 or self.num_workers == 1:
            P, Q, T = _binary_split(0, n)
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(_worker_binary_split, chunk): i 
                          for i, chunk in enumerate(chunks)}
                
                results = [None] * len(chunks)
                for future in as_completed(futures):
                    idx = futures[future]
                    P_bytes, Q_bytes, T_bytes = future.result()
                    # Convert back from bytes
                    results[idx] = (
                        gmpy2.from_binary(P_bytes),
                        gmpy2.from_binary(Q_bytes),
                        gmpy2.from_binary(T_bytes)
                    )
            
            # Merge results in order
            while len(results) > 1:
                new_results = []
                for i in range(0, len(results), 2):
                    if i + 1 < len(results):
                        merged = _merge_results(results[i], results[i+1])
                        new_results.append(merged)
                    else:
                        new_results.append(results[i])
                results = new_results
            
            P, Q, T = results[0]
        
        # Final computation: π = sqrt(C³) × Q / (12 × T)
        sqrt_C3 = gmpy2.sqrt(mpfr(C3))
        pi = sqrt_C3 * mpfr(Q) / (12 * mpfr(T))
        
        pi_str = str(pi)
        if '.' in pi_str:
            return pi_str.split('.')[1][:self.precision]
        return pi_str[:self.precision]


def verify_digits(digits: str, n: int = 50) -> bool:
    """Verify first n digits of π."""
    PI_FIRST_50 = "14159265358979323846264338327950288419716939937510"
    return digits[:n] == PI_FIRST_50[:n]


def benchmark():
    """Benchmark parallel vs sequential Chudnovsky."""
    print("=" * 70)
    print("PARALLEL CHUDNOVSKY BENCHMARK")
    print("=" * 70)
    print(f"CPU cores available: {mp.cpu_count()}")
    
    # Test different precisions
    precisions = [100000, 500000, 1000000, 2000000]
    
    print("\n[1] Sequential (1 worker)")
    print("-" * 50)
    
    seq_times = {}
    for prec in precisions:
        pc = ParallelChudnovsky(prec, num_workers=1)
        
        t0 = time.time()
        digits = pc.compute()
        elapsed = time.time() - t0
        
        seq_times[prec] = elapsed
        verified = "✓" if verify_digits(digits) else "✗"
        rate = prec / elapsed
        
        print(f"  {prec:>10,} digits: {elapsed:.2f}s ({rate:>12,.0f} d/s) {verified}")
    
    print(f"\n[2] Parallel ({mp.cpu_count()} workers)")
    print("-" * 50)
    
    for prec in precisions:
        pc = ParallelChudnovsky(prec)  # Uses all cores
        
        t0 = time.time()
        digits = pc.compute()
        elapsed = time.time() - t0
        
        verified = "✓" if verify_digits(digits) else "✗"
        rate = prec / elapsed
        speedup = seq_times[prec] / elapsed
        
        print(f"  {prec:>10,} digits: {elapsed:.2f}s ({rate:>12,.0f} d/s) {verified} [{speedup:.1f}x]")
    
    print("\n[3] Scaling Test (1M digits)")
    print("-" * 50)
    
    prec = 1000000
    for n_workers in [1, 2, 4, 8, 12]:
        if n_workers > mp.cpu_count():
            continue
            
        pc = ParallelChudnovsky(prec, num_workers=n_workers)
        
        t0 = time.time()
        digits = pc.compute()
        elapsed = time.time() - t0
        
        verified = "✓" if verify_digits(digits) else "✗"
        rate = prec / elapsed
        speedup = seq_times.get(prec, elapsed) / elapsed
        
        print(f"  {n_workers:>2} workers: {elapsed:.2f}s ({rate:>10,.0f} d/s) {verified} [{speedup:.2f}x]")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    benchmark()
