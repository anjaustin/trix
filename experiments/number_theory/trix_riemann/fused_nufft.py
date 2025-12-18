#!/usr/bin/env python3
"""
FUSED NUFFT - Single Kernel Riemann-Siegel
===========================================

Fuses ALL operations into minimal kernel launches:
1. Coefficient generation + B-spline spreading → ONE kernel
2. FFT (already fused in cuFFT)
3. Theta + final Z → ONE kernel

Target: 100x speedup by eliminating Python overhead.
"""

import torch
import triton
import triton.language as tl
import math
import time

PI = math.pi
TWO_PI = 2 * PI


@triton.jit
def fused_spread_kernel(
    # Outputs
    grid_re_ptr, grid_im_ptr,
    # Inputs
    n_ptr,          # n values: 1, 2, 3, ..., M
    ln_n_ptr,       # log(n) precomputed
    rsqrt_n_ptr,    # 1/sqrt(n) precomputed
    # Parameters
    t0,             # Starting t
    delta,          # t spacing
    delta_omega,    # Grid spacing in frequency
    M: tl.constexpr,
    num_grid: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: compute coefficients AND spread to grid.
    
    For each n:
      1. a_n = rsqrt(n) * exp(i * t0 * ln(n))
      2. omega_n = delta * ln(n)
      3. grid_idx = omega_n / delta_omega
      4. Spread using cubic B-spline to 4 grid points
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M
    
    # Load precomputed values
    n = tl.load(n_ptr + offs, mask=mask, other=1.0)
    ln_n = tl.load(ln_n_ptr + offs, mask=mask, other=0.0)
    rsqrt_n = tl.load(rsqrt_n_ptr + offs, mask=mask, other=0.0)
    
    # Coefficient: a_n = rsqrt(n) * exp(i * t0 * ln(n))
    phase = t0 * ln_n
    a_re = rsqrt_n * tl.cos(phase)
    a_im = rsqrt_n * tl.sin(phase)
    
    # Grid index (fractional)
    omega_n = delta * ln_n
    grid_idx = omega_n / delta_omega
    idx_floor = grid_idx.to(tl.int32)
    frac = grid_idx - idx_floor.to(tl.float64)
    
    # Cubic B-spline weights
    t = frac
    t2 = t * t
    t3 = t2 * t
    w0 = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
    w1 = (3.0*t3 - 6.0*t2 + 4.0) / 6.0
    w2 = (-3.0*t3 + 3.0*t2 + 3.0*t + 1.0) / 6.0
    w3 = t3 / 6.0
    
    # Scatter to 4 grid points (atomic add)
    # Handle negative modulo correctly: ((x % n) + n) % n
    # Point 0: idx_floor - 1
    idx0_raw = idx_floor - 1
    idx0 = ((idx0_raw % num_grid) + num_grid) % num_grid
    valid0 = mask & (idx0 >= 0) & (idx0 < num_grid)
    tl.atomic_add(grid_re_ptr + idx0, a_re * w0, mask=valid0)
    tl.atomic_add(grid_im_ptr + idx0, a_im * w0, mask=valid0)
    
    # Point 1: idx_floor
    idx1 = ((idx_floor % num_grid) + num_grid) % num_grid
    valid1 = mask & (idx1 >= 0) & (idx1 < num_grid)
    tl.atomic_add(grid_re_ptr + idx1, a_re * w1, mask=valid1)
    tl.atomic_add(grid_im_ptr + idx1, a_im * w1, mask=valid1)
    
    # Point 2: idx_floor + 1
    idx2 = ((idx_floor + 1) % num_grid + num_grid) % num_grid
    valid2 = mask & (idx2 >= 0) & (idx2 < num_grid)
    tl.atomic_add(grid_re_ptr + idx2, a_re * w2, mask=valid2)
    tl.atomic_add(grid_im_ptr + idx2, a_im * w2, mask=valid2)
    
    # Point 3: idx_floor + 2
    idx3 = ((idx_floor + 2) % num_grid + num_grid) % num_grid
    valid3 = mask & (idx3 >= 0) & (idx3 < num_grid)
    tl.atomic_add(grid_re_ptr + idx3, a_re * w3, mask=valid3)
    tl.atomic_add(grid_im_ptr + idx3, a_im * w3, mask=valid3)


@triton.jit
def fused_theta_z_kernel(
    # Outputs
    Z_ptr,
    # Inputs
    S_re_ptr, S_im_ptr,  # FFT result (main sum)
    t_ptr,               # t values
    # Parameters
    num_t: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: compute theta and final Z.
    
    Z(t) = 2 * Re(S(t) * exp(-i * theta(t)))
    
    theta(t) = t/2 * ln(t/2π) - t/2 - π/8 + 1/(48t)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_t
    
    # Load
    S_re = tl.load(S_re_ptr + offs, mask=mask, other=0.0)
    S_im = tl.load(S_im_ptr + offs, mask=mask, other=0.0)
    t = tl.load(t_ptr + offs, mask=mask, other=1.0)
    
    # Theta
    LOG_2PI: tl.constexpr = 1.8378770664093453
    PI: tl.constexpr = 3.141592653589793
    
    t_half = t / 2.0
    theta = t_half * (tl.log(t) - LOG_2PI) - t_half - PI / 8.0
    theta = theta + 1.0 / (48.0 * t)
    
    # exp(-i * theta)
    cos_th = tl.cos(theta)
    sin_th = tl.sin(theta)
    
    # S * exp(-i*theta) = (S_re + i*S_im) * (cos - i*sin)
    #                   = S_re*cos + S_im*sin + i*(S_im*cos - S_re*sin)
    result_re = S_re * cos_th + S_im * sin_th
    
    # Z = 2 * Re(...)
    Z = 2.0 * result_re
    
    tl.store(Z_ptr + offs, Z, mask=mask)


class FusedNUFFT:
    """
    Fused NUFFT for Riemann zero hunting.
    
    Minimal kernel launches:
    1. fused_spread_kernel (coefficients + B-spline spreading)
    2. torch.fft.ifft (cuFFT, already optimal)
    3. fused_theta_z_kernel (theta + final Z)
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def evaluate_Z(self, t0: float, num_evals: int, oversampling: int = 4):
        """
        Evaluate Z(t) at num_evals equally-spaced points starting at t0.
        """
        device = self.device
        
        # Parameters
        M = int(math.sqrt(t0 / TWO_PI)) + 10
        num_grid = num_evals * oversampling
        
        # Delta: spacing in t
        density = math.log(t0) / TWO_PI
        delta = 1.0 / (density * 10)  # 10 points per expected zero
        
        delta_omega = TWO_PI / num_grid
        
        # Precompute n, ln(n), rsqrt(n)
        n = torch.arange(1, M + 1, device=device, dtype=torch.float64)
        ln_n = torch.log(n)
        rsqrt_n = torch.rsqrt(n)
        
        # Allocate grid (real and imag separate for Triton)
        grid_re = torch.zeros(num_grid, device=device, dtype=torch.float64)
        grid_im = torch.zeros(num_grid, device=device, dtype=torch.float64)
        
        # Launch spread kernel
        BLOCK_SIZE = 256
        num_blocks = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        fused_spread_kernel[(num_blocks,)](
            grid_re, grid_im,
            n, ln_n, rsqrt_n,
            t0, delta, delta_omega,
            M, num_grid,
            BLOCK_SIZE,
        )
        
        # FFT (use torch.fft on combined complex)
        grid = torch.complex(grid_re, grid_im)
        G = torch.fft.ifft(grid) * num_grid
        
        # Extract S values
        S_re = G.real[:num_evals].contiguous()
        S_im = G.imag[:num_evals].contiguous()
        
        # t values
        t_vals = t0 + delta * torch.arange(num_evals, device=device, dtype=torch.float64)
        
        # Allocate output
        Z = torch.empty(num_evals, device=device, dtype=torch.float64)
        
        # Launch theta+Z kernel
        num_blocks_z = (num_evals + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        fused_theta_z_kernel[(num_blocks_z,)](
            Z,
            S_re, S_im,
            t_vals,
            num_evals,
            BLOCK_SIZE,
        )
        
        return t_vals.float(), Z.float(), M


def benchmark():
    """Benchmark fused vs unfused."""
    print("="*70)
    print("FUSED NUFFT BENCHMARK")
    print("="*70)
    
    device = 'cuda'
    fused = FusedNUFFT(device)
    
    # Compare at different scales
    for t0 in [1e6, 1e7, 1e8, 1e9]:
        M = int(math.sqrt(t0 / TWO_PI)) + 10
        num_evals = M * 2
        
        # Warmup
        for _ in range(3):
            fused.evaluate_Z(t0, num_evals)
        torch.cuda.synchronize()
        
        # Time
        start = time.time()
        for _ in range(100):
            t_vals, Z_vals, _ = fused.evaluate_Z(t0, num_evals)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 100
        
        # Count zeros
        signs = torch.sign(Z_vals)
        zeros = ((signs[:-1] * signs[1:]) < 0).sum().item()
        
        zeros_per_sec = zeros / elapsed
        evals_per_sec = num_evals / elapsed
        
        print(f"t={t0:.0e}: M={M:>6}, {elapsed*1000:.2f}ms, "
              f"{zeros_per_sec:>12,.0f} zeros/s, {evals_per_sec:>12,.0f} evals/s")
    
    print()
    print("="*70)


if __name__ == "__main__":
    benchmark()
