#!/usr/bin/env python3
"""
Hollywood Gather NUFFT
======================

Key insight: Scatter → Gather transformation eliminates atomics.

Instead of each source writing to 4 destinations (collisions, atomics),
each destination reads from its sources (no collisions, pure parallel).

Source ranges are CONTIGUOUS in n, enabling coalesced memory access.
"""

import torch
import triton
import triton.language as tl
import math
import time

PI = math.pi
TWO_PI = 2 * PI


def cubic_bspline_torch(x):
    """Cubic B-spline kernel (width 4)."""
    ax = torch.abs(x)
    result = torch.zeros_like(x)
    
    mask1 = ax < 1
    result[mask1] = (2/3) - ax[mask1]**2 + 0.5 * ax[mask1]**3
    
    mask2 = (ax >= 1) & (ax < 2)
    result[mask2] = (1/6) * (2 - ax[mask2])**3
    
    return result


def build_gather_map(M, grid_idx, num_grid, device='cuda'):
    """
    Build inverse mapping: for each grid point, which sources contribute?
    
    Returns:
        n_lo: start of source range for each grid point
        n_hi: end of source range (exclusive)
        max_active: highest active grid index
    """
    # Find max active grid index
    max_gidx = grid_idx[-1].item()
    max_active = min(num_grid, int(max_gidx) + 3)
    
    n_lo = torch.zeros(max_active, device=device, dtype=torch.int32)
    n_hi = torch.zeros(max_active, device=device, dtype=torch.int32)
    
    # For each grid point m, find n range where |grid_idx[n] - m| < 2
    # Since grid_idx is monotonic increasing, we can binary search
    
    grid_idx_cpu = grid_idx.cpu().numpy()
    
    for m in range(max_active):
        # Find n where grid_idx[n-1] >= m - 2 (lower bound)
        # and grid_idx[n-1] < m + 2 (upper bound)
        lo_target = m - 2
        hi_target = m + 2
        
        # Binary search for lower bound
        lo = 0
        hi = M
        while lo < hi:
            mid = (lo + hi) // 2
            if grid_idx_cpu[mid] < lo_target:
                lo = mid + 1
            else:
                hi = mid
        n_start = lo
        
        # Binary search for upper bound
        lo = 0
        hi = M
        while lo < hi:
            mid = (lo + hi) // 2
            if grid_idx_cpu[mid] < hi_target:
                lo = mid + 1
            else:
                hi = mid
        n_end = lo
        
        n_lo[m] = n_start
        n_hi[m] = n_end
    
    return n_lo, n_hi, max_active


def gather_spread_python(a_n, grid_idx, n_lo, n_hi, max_active, num_grid, device='cuda'):
    """
    Gather-based spreading (Python reference implementation).
    
    Each grid point independently sums its contributing sources.
    No atomics needed!
    """
    grid = torch.zeros(num_grid, device=device, dtype=a_n.dtype)
    
    for m in range(max_active):
        lo, hi = n_lo[m].item(), n_hi[m].item()
        if hi > lo:
            ns = torch.arange(lo, hi, device=device)
            gidx = grid_idx[ns]
            weights = cubic_bspline_torch(gidx - m)
            grid[m] = (a_n[ns] * weights).sum()
    
    return grid


@triton.jit
def gather_spread_kernel(
    # Output
    grid_re_ptr, grid_im_ptr,
    # Input  
    a_re_ptr, a_im_ptr,
    grid_idx_ptr,
    n_lo_ptr, n_hi_ptr,
    # Sizes
    max_active,
    num_grid,
    BLOCK_SIZE: tl.constexpr,
    MAX_RANGE: tl.constexpr,
):
    """
    Triton kernel for gather-based spreading.
    
    Each program handles one grid point.
    Loads contiguous range of sources, computes weights, sums.
    """
    m = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_m = m < max_active
    
    # Load source range for this grid point
    lo = tl.load(n_lo_ptr + m, mask=mask_m, other=0)
    hi = tl.load(n_hi_ptr + m, mask=mask_m, other=0)
    
    # Initialize accumulators
    sum_re = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    sum_im = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    
    # Process sources in chunks
    for offset in range(0, MAX_RANGE):
        n_idx = lo + offset
        mask_n = mask_m & (n_idx < hi)
        
        # Load source values
        a_re = tl.load(a_re_ptr + n_idx, mask=mask_n, other=0.0)
        a_im = tl.load(a_im_ptr + n_idx, mask=mask_n, other=0.0)
        gidx = tl.load(grid_idx_ptr + n_idx, mask=mask_n, other=0.0)
        
        # B-spline weight
        dist = gidx - m.to(tl.float64)
        ax = tl.abs(dist)
        
        # Cubic B-spline
        w = tl.where(ax < 1.0,
                     (2.0/3.0) - ax*ax + 0.5*ax*ax*ax,
                     tl.where(ax < 2.0,
                              (1.0/6.0) * (2.0 - ax) * (2.0 - ax) * (2.0 - ax),
                              0.0))
        
        # Accumulate
        sum_re += tl.where(mask_n, a_re * w, 0.0)
        sum_im += tl.where(mask_n, a_im * w, 0.0)
    
    # Store results
    tl.store(grid_re_ptr + m, sum_re, mask=mask_m)
    tl.store(grid_im_ptr + m, sum_im, mask=mask_m)


class HollywoodGatherNUFFT:
    """
    Hollywood-style NUFFT using gather instead of scatter.
    
    Pre-computes the inverse mapping once, then uses it for all evaluations.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.gather_map = None
        self.cached_params = None
    
    def evaluate_Z(self, t0: float, num_evals: int, oversampling: int = 4):
        """Evaluate Z(t) using gather-based NUFFT."""
        device = self.device
        
        # Parameters
        M = int(math.sqrt(t0 / TWO_PI)) + 10
        num_grid = num_evals * oversampling
        
        density = math.log(t0) / TWO_PI
        delta = 1.0 / (density * 10)
        delta_omega = TWO_PI / num_grid
        
        # Precompute source data
        n = torch.arange(1, M + 1, device=device, dtype=torch.float64)
        ln_n = torch.log(n)
        rsqrt_n = torch.rsqrt(n)
        
        # Coefficients
        phase = t0 * ln_n
        a_re = rsqrt_n * torch.cos(phase)
        a_im = rsqrt_n * torch.sin(phase)
        
        # Grid indices (fractional)
        grid_idx = delta * ln_n / delta_omega
        
        # Build or reuse gather map
        params = (M, num_grid, delta, delta_omega)
        if self.gather_map is None or self.cached_params != params:
            n_lo, n_hi, max_active = build_gather_map(M, grid_idx, num_grid, device)
            self.gather_map = (n_lo, n_hi, max_active)
            self.cached_params = params
        else:
            n_lo, n_hi, max_active = self.gather_map
        
        # Allocate grid
        grid_re = torch.zeros(num_grid, device=device, dtype=torch.float64)
        grid_im = torch.zeros(num_grid, device=device, dtype=torch.float64)
        
        # Determine max range size
        max_range = (n_hi - n_lo).max().item()
        
        # Launch gather kernel
        BLOCK_SIZE = 1
        num_blocks = max_active
        
        # Use Python fallback for now (Triton kernel needs tuning)
        grid = gather_spread_python(
            torch.complex(a_re, a_im), grid_idx, n_lo, n_hi, max_active, num_grid, device
        )
        
        # FFT
        G = torch.fft.ifft(grid) * num_grid
        S = G[:num_evals]
        
        # Theta and Z
        t_vals = t0 + delta * torch.arange(num_evals, device=device, dtype=torch.float64)
        th = t_vals/2 * torch.log(t_vals/TWO_PI) - t_vals/2 - PI/8 + 1/(48*t_vals)
        Z = 2.0 * (S * torch.exp(-1j * th)).real
        
        return t_vals.float(), Z.float(), M


def benchmark():
    """Compare gather vs scatter approaches."""
    print("="*70)
    print("HOLLYWOOD GATHER vs SCATTER BENCHMARK")
    print("="*70)
    
    device = 'cuda'
    
    # Import scatter-based for comparison
    from fused_nufft import FusedNUFFT
    scatter_nufft = FusedNUFFT(device)
    gather_nufft = HollywoodGatherNUFFT(device)
    
    for t0 in [1e7, 1e8, 1e9]:
        M = int(math.sqrt(t0 / TWO_PI)) + 10
        num_evals = M * 8
        
        print(f"\nt0={t0:.0e}, M={M}, evals={num_evals}")
        
        # Warmup
        for _ in range(3):
            scatter_nufft.evaluate_Z(t0, num_evals)
            gather_nufft.evaluate_Z(t0, num_evals)
        torch.cuda.synchronize()
        
        # Benchmark scatter
        start = time.time()
        for _ in range(20):
            _, Z_scatter, _ = scatter_nufft.evaluate_Z(t0, num_evals)
        torch.cuda.synchronize()
        scatter_time = (time.time() - start) / 20
        
        # Benchmark gather  
        start = time.time()
        for _ in range(20):
            _, Z_gather, _ = gather_nufft.evaluate_Z(t0, num_evals)
        torch.cuda.synchronize()
        gather_time = (time.time() - start) / 20
        
        # Compare
        zeros_scatter = ((torch.sign(Z_scatter[:-1]) * torch.sign(Z_scatter[1:])) < 0).sum().item()
        zeros_gather = ((torch.sign(Z_gather[:-1]) * torch.sign(Z_gather[1:])) < 0).sum().item()
        
        print(f"  Scatter: {scatter_time*1000:.2f}ms, {zeros_scatter} zeros, "
              f"{zeros_scatter/scatter_time/1e6:.2f}M zeros/s")
        print(f"  Gather:  {gather_time*1000:.2f}ms, {zeros_gather} zeros, "
              f"{zeros_gather/gather_time/1e6:.2f}M zeros/s")
        print(f"  Speedup: {scatter_time/gather_time:.2f}x")
        print(f"  Zero agreement: {min(zeros_scatter, zeros_gather)/max(zeros_scatter, zeros_gather)*100:.1f}%")


if __name__ == "__main__":
    benchmark()
