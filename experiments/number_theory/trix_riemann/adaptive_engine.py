#!/usr/bin/env python3
"""
ADAPTIVE PRECISION RIEMANN ENGINE
=================================

Automatically switches precision based on scale:
- t < 10^12: Hybrid (FP64 phases, FP32 FFT) - faster
- t >= 10^12: Pure FP64 - more accurate, less overhead at scale

Peak rates:
- Hybrid: ~22M zeros/sec at t=10^10
- FP64: ~5M zeros/sec at t=10^10, but scales better
"""

import torch
import math

PI = math.pi
TWO_PI = 2 * PI


class HybridEngine:
    """FP32 FFT with FP64 phase reduction. Fast for t < 10^12."""
    
    def __init__(self, t0_center, num_evals, device='cuda'):
        self.device = device
        self.num_evals = num_evals
        
        self.M = int(math.sqrt(t0_center / TWO_PI)) + 10
        self.num_grid = num_evals * 4
        
        density = math.log(t0_center) / TWO_PI
        self.delta = 1.0 / (density * 10)
        
        n = torch.arange(1, self.M + 1, device=device, dtype=torch.float64)
        self.ln_n = torch.log(n)
        self.rsqrt_n = torch.rsqrt(n).float()
        
        grid_idx = (self.delta * self.ln_n / (TWO_PI / self.num_grid)).float()
        idx_floor = grid_idx.long()
        frac = grid_idx - idx_floor.float()
        
        offsets = torch.tensor([-1, 0, 1, 2], device=device)
        self.scatter_indices = ((idx_floor.unsqueeze(1) + offsets.unsqueeze(0)) % self.num_grid).flatten().long()
        
        t = frac
        self.weights = torch.stack([
            (1 - t)**3 / 6,
            (3*t**3 - 6*t**2 + 4) / 6,
            (-3*t**3 + 3*t**2 + 3*t + 1) / 6,
            t**3 / 6
        ], dim=1).to(torch.complex64)
        
        self.k = torch.arange(num_evals, device=device, dtype=torch.float32)
    
    def evaluate(self, t0):
        phase = (t0 * self.ln_n) % TWO_PI
        a_n = torch.complex(
            self.rsqrt_n * torch.cos(phase).float(),
            self.rsqrt_n * torch.sin(phase).float()
        )
        
        contributions = (a_n.unsqueeze(1) * self.weights).flatten()
        grid = torch.zeros(self.num_grid, device=self.device, dtype=torch.complex64)
        grid.scatter_add_(0, self.scatter_indices, contributions)
        
        G = torch.fft.ifft(grid) * self.num_grid
        S = G[:self.num_evals]
        
        t_vals = t0 + self.delta * self.k.double()
        th = (t_vals/2 * torch.log(t_vals/TWO_PI) - t_vals/2 - PI/8 + 1/(48*t_vals)) % TWO_PI
        
        Z = 2.0 * (S.real * torch.cos(th).float() + S.imag * torch.sin(th).float())
        
        return t_vals.float(), Z


class FP64Engine:
    """Pure FP64. Better for t >= 10^12."""
    
    def __init__(self, t0_center, num_evals, device='cuda'):
        self.device = device
        self.num_evals = num_evals
        
        self.M = int(math.sqrt(t0_center / TWO_PI)) + 10
        self.num_grid = num_evals * 4
        
        density = math.log(t0_center) / TWO_PI
        self.delta = 1.0 / (density * 10)
        
        n = torch.arange(1, self.M + 1, device=device, dtype=torch.float64)
        self.ln_n = torch.log(n)
        self.rsqrt_n = torch.rsqrt(n)
        
        grid_idx = self.delta * self.ln_n / (TWO_PI / self.num_grid)
        idx_floor = grid_idx.long()
        frac = grid_idx - idx_floor.float()
        
        offsets = torch.tensor([-1, 0, 1, 2], device=device)
        self.scatter_indices = ((idx_floor.unsqueeze(1) + offsets.unsqueeze(0)) % self.num_grid).flatten().long()
        
        t = frac
        self.weights = torch.stack([
            (1 - t)**3 / 6,
            (3*t**3 - 6*t**2 + 4) / 6,
            (-3*t**3 + 3*t**2 + 3*t + 1) / 6,
            t**3 / 6
        ], dim=1).to(torch.complex128)
        
        self.k = torch.arange(num_evals, device=device, dtype=torch.float64)
    
    def evaluate(self, t0):
        phase = t0 * self.ln_n
        a_n = (self.rsqrt_n * torch.exp(1j * phase)).to(torch.complex128)
        
        contributions = (a_n.unsqueeze(1) * self.weights).flatten()
        grid = torch.zeros(self.num_grid, device=self.device, dtype=torch.complex128)
        grid.scatter_add_(0, self.scatter_indices, contributions)
        
        G = torch.fft.ifft(grid) * self.num_grid
        S = G[:self.num_evals]
        
        t_vals = t0 + self.delta * self.k
        th = t_vals/2 * torch.log(t_vals/TWO_PI) - t_vals/2 - PI/8 + 1/(48*t_vals)
        Z = 2.0 * (S * torch.exp(-1j * th)).real
        
        return t_vals.float(), Z.float()


class AdaptiveEngine:
    """
    Automatically selects precision based on t.
    
    - t < 10^12: Hybrid (FP32 FFT) - ~4x faster
    - t >= 10^12: FP64 - scales better, more accurate
    """
    
    HYBRID_THRESHOLD = 1e12
    
    def __init__(self, t0_center, num_evals, device='cuda'):
        self.device = device
        self.t0_center = t0_center
        self.num_evals = num_evals
        
        if t0_center < self.HYBRID_THRESHOLD:
            self.engine = HybridEngine(t0_center, num_evals, device)
            self.mode = 'hybrid'
        else:
            self.engine = FP64Engine(t0_center, num_evals, device)
            self.mode = 'fp64'
        
        self.M = self.engine.M
        self.delta = self.engine.delta
    
    def evaluate(self, t0):
        return self.engine.evaluate(t0)
    
    def rebuild_if_needed(self, t0):
        """Rebuild engine if t has grown significantly or crossed threshold."""
        # Check if we need to switch modes
        was_hybrid = self.mode == 'hybrid'
        should_be_hybrid = t0 < self.HYBRID_THRESHOLD
        
        # Check if M has grown significantly
        new_M = int(math.sqrt(t0 / TWO_PI)) + 10
        M_ratio = new_M / self.M
        
        if (was_hybrid != should_be_hybrid) or (M_ratio > 2):
            self.__init__(t0, self.num_evals, self.device)
            return True
        return False


def benchmark():
    """Benchmark adaptive engine across scales."""
    import time
    
    print('='*70)
    print('ADAPTIVE ENGINE BENCHMARK')
    print('='*70)
    print()
    
    device = 'cuda'
    
    print(f'{"t0":>12} {"mode":>8} {"M":>8} {"time":>10} {"zeros":>10} {"rate":>15}')
    print('-'*70)
    
    for t0 in [1e8, 1e9, 1e10, 1e11, 1e12, 1e13]:
        M = int(math.sqrt(t0 / TWO_PI)) + 10
        num_evals = M * 8
        
        engine = AdaptiveEngine(t0, num_evals, device)
        
        # Warmup
        for _ in range(3):
            engine.evaluate(t0)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        iters = max(5, 50000 // num_evals)
        for _ in range(iters):
            t_vals, Z_vals = engine.evaluate(t0)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / iters
        
        zeros = ((torch.sign(Z_vals[:-1]) * torch.sign(Z_vals[1:])) < 0).sum().item()
        rate = zeros / elapsed
        
        print(f'{t0:>12.0e} {engine.mode:>8} {M:>8} {elapsed*1000:>8.2f}ms {zeros:>10} {rate/1e6:>12.2f}M/s')
    
    print('='*70)


if __name__ == '__main__':
    benchmark()
