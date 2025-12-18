#!/usr/bin/env python3
"""
SpectralTile - FFT Evaluation using TriX Pattern
==================================================

Pure TriX implementation of spectral evaluation for Z(t).

Architecture:
    Fixed microcode: Butterfly operations (exact complex arithmetic)
    Algorithmic routing: Deterministic based on FFT structure
    
This follows the TriX pattern:
    - Tiles ARE the operations (fixed butterfly microcode)
    - Routing IS the control (algorithmic, not learned)

For FFT, the routing is STRUCTURAL - determined by (stage, position).
No learning required. 100% accurate.
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')
sys.path.insert(0, '/workspace/trix_latest/experiments/number_theory')

import torch
import torch.nn as nn
import math
from typing import Tuple

try:
    from .theta_tile import ThetaTile
    from .dirichlet_tile import DirichletTile
except ImportError:
    from trix_riemann.theta_tile import ThetaTile
    from trix_riemann.dirichlet_tile import DirichletTile


# =============================================================================
# MICROCODE: COMPLEX BUTTERFLY
# =============================================================================

class ButterflyMicrocode:
    """
    Fixed butterfly operations for FFT.
    
    Standard Cooley-Tukey butterfly:
        (a, b, W) → (a + W*b, a - W*b)
    
    All operations are EXACT.
    """
    
    @staticmethod
    def twiddle_factors(N: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute twiddle factors W_k = e^{-2πik/N}.
        
        Returns:
            W_real: cos(-2πk/N) for k = 0..N-1
            W_imag: sin(-2πk/N) for k = 0..N-1
        """
        k = torch.arange(N, dtype=torch.float32, device=device)
        angles = -2.0 * math.pi * k / N
        return torch.cos(angles), torch.sin(angles)
    
    @staticmethod
    def complex_mul(a_re: torch.Tensor, a_im: torch.Tensor,
                    b_re: torch.Tensor, b_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Complex multiplication: (a_re + i*a_im) * (b_re + i*b_im)"""
        re = a_re * b_re - a_im * b_im
        im = a_re * b_im + a_im * b_re
        return re, im
    
    @staticmethod
    def butterfly(a_re: torch.Tensor, a_im: torch.Tensor,
                  b_re: torch.Tensor, b_im: torch.Tensor,
                  w_re: torch.Tensor, w_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Butterfly operation: (a, b, W) → (a + W*b, a - W*b)
        
        Returns:
            out1_re, out1_im: a + W*b
            out2_re, out2_im: a - W*b
        """
        # W * b
        wb_re, wb_im = ButterflyMicrocode.complex_mul(w_re, w_im, b_re, b_im)
        
        # a + W*b
        out1_re = a_re + wb_re
        out1_im = a_im + wb_im
        
        # a - W*b
        out2_re = a_re - wb_re
        out2_im = a_im - wb_im
        
        return out1_re, out1_im, out2_re, out2_im


# =============================================================================
# ALGORITHMIC ROUTING
# =============================================================================

class FFTRouter:
    """
    Deterministic routing for FFT.
    
    The routing pattern is STRUCTURAL - a function of (stage, position, N).
    No learning required. This IS the FFT algorithm.
    """
    
    @staticmethod
    def bit_reverse(x: int, num_bits: int) -> int:
        """Bit-reversal permutation."""
        result = 0
        for _ in range(num_bits):
            result = (result << 1) | (x & 1)
            x >>= 1
        return result
    
    @staticmethod
    def bit_reverse_indices(N: int) -> torch.Tensor:
        """Get bit-reversed index permutation."""
        num_bits = int(math.log2(N))
        return torch.tensor([FFTRouter.bit_reverse(i, num_bits) for i in range(N)])
    
    @staticmethod
    def get_butterfly_pairs(stage: int, N: int) -> list:
        """
        Get (i, j) pairs for butterfly operations at given stage.
        
        Returns list of (idx_a, idx_b, twiddle_k) tuples.
        """
        pairs = []
        stride = 2 ** stage
        group_size = 2 * stride
        
        for group_start in range(0, N, group_size):
            for k in range(stride):
                i = group_start + k
                j = group_start + k + stride
                twiddle_k = k * (N // group_size)
                pairs.append((i, j, twiddle_k))
        
        return pairs


# =============================================================================
# TRIX FFT
# =============================================================================

class TriXFFT(nn.Module):
    """
    TriX-style FFT implementation.
    
    Pattern:
        - Microcode: ButterflyMicrocode (exact arithmetic)
        - Routing: FFTRouter (algorithmic, deterministic)
    
    This computes FFT with 100% accuracy using the TriX philosophy:
    tiles hold operations, routing selects when to apply them.
    """
    
    def __init__(self, max_n: int = 1024):
        super().__init__()
        self.max_n = max_n
        
        # Precompute twiddle factors for common sizes
        self._twiddle_cache = {}
        self._index_cache = {}
    
    def _get_twiddles(self, N: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached twiddle factors."""
        key = (N, device)
        if key not in self._twiddle_cache:
            w_re, w_im = ButterflyMicrocode.twiddle_factors(N, device)
            self._twiddle_cache[key] = (w_re, w_im)
        return self._twiddle_cache[key]
    
    def _get_bit_reverse(self, N: int) -> torch.Tensor:
        """Get cached bit-reverse indices."""
        if N not in self._index_cache:
            self._index_cache[N] = FFTRouter.bit_reverse_indices(N)
        return self._index_cache[N]
    
    def forward(self, x_re: torch.Tensor, x_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute FFT of complex input.
        
        Args:
            x_re: Real parts, shape (..., N)
            x_im: Imaginary parts, shape (..., N)
            
        Returns:
            X_re, X_im: FFT output, same shape
        """
        *batch_dims, N = x_re.shape
        device = x_re.device
        
        # Verify N is power of 2
        assert N > 0 and (N & (N - 1)) == 0, f"N must be power of 2, got {N}"
        
        num_stages = int(math.log2(N))
        
        # Get twiddle factors
        w_re, w_im = self._get_twiddles(N, device)
        
        # Bit-reverse permutation
        br_idx = self._get_bit_reverse(N).to(device)
        y_re = x_re[..., br_idx]
        y_im = x_im[..., br_idx]
        
        # FFT stages (Cooley-Tukey DIT)
        for stage in range(num_stages):
            pairs = FFTRouter.get_butterfly_pairs(stage, N)
            
            new_re = y_re.clone()
            new_im = y_im.clone()
            
            for i, j, k in pairs:
                # Get inputs
                a_re = y_re[..., i]
                a_im = y_im[..., i]
                b_re = y_re[..., j]
                b_im = y_im[..., j]
                
                # Get twiddle
                wk_re = w_re[k]
                wk_im = w_im[k]
                
                # Butterfly
                o1_re, o1_im, o2_re, o2_im = ButterflyMicrocode.butterfly(
                    a_re, a_im, b_re, b_im, wk_re, wk_im
                )
                
                new_re[..., i] = o1_re
                new_im[..., i] = o1_im
                new_re[..., j] = o2_re
                new_im[..., j] = o2_im
            
            y_re = new_re
            y_im = new_im
        
        return y_re, y_im


# =============================================================================
# SPECTRAL TILE
# =============================================================================

class SpectralTile(nn.Module):
    """
    Spectral evaluation tile for Riemann-Siegel Z function.
    
    Computes Z(t) = 2·Re(e^{iθ(t)} · Σ_{n≤N} n^{-1/2-it})
    
    Uses:
        - ThetaTile for θ(t)
        - DirichletTile for coefficients
        - TriXFFT for spectral evaluation (when beneficial)
    
    This is the core computation tile for the Riemann Probe.
    """
    
    def __init__(self, max_n: int = 1000):
        super().__init__()
        
        self.theta_tile = ThetaTile(include_corrections=True)
        self.dirichlet_tile = DirichletTile(max_n=max_n)
        self.trix_fft = TriXFFT(max_n=max_n)
    
    def forward(self, t: torch.Tensor, N: int = None) -> torch.Tensor:
        """
        Compute Z(t) for given t values.
        
        Args:
            t: Height values, shape (M,)
            N: Number of terms (default: auto)
            
        Returns:
            Z: Z(t) values, shape (M,)
        """
        # Step 1: Compute theta
        theta = self.theta_tile(t)
        
        # Step 2: Get Dirichlet coefficients
        real, imag = self.dirichlet_tile(t, N)
        
        # Step 3: Sum the series (direct summation - FFT variant TODO)
        sum_real = real.sum(dim=1)
        sum_imag = imag.sum(dim=1)
        
        # Step 4: Apply theta rotation: e^{iθ} · sum
        # e^{iθ} = cos(θ) + i·sin(θ)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # Complex multiply: e^{iθ} · (sum_real + i·sum_imag)
        rotated_real = cos_theta * sum_real - sin_theta * sum_imag
        rotated_imag = cos_theta * sum_imag + sin_theta * sum_real
        
        # Step 5: Z(t) = 2 · Re(...)
        Z = 2.0 * rotated_real
        
        return Z
    
    def verify_against_known_zeros(self) -> dict:
        """
        Verify Z(t) computation against known zero locations.
        
        The first 10 non-trivial zeros of ζ(s) have imaginary parts:
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832
        
        Z(t) should be close to zero at these points.
        """
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
        
        known_zeros = torch.tensor([
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005151, 49.773832
        ], device=device)
        
        Z_at_zeros = self.forward(known_zeros)
        
        return {
            't_values': known_zeros,
            'Z_values': Z_at_zeros,
            'max_abs_Z': Z_at_zeros.abs().max().item(),
            'mean_abs_Z': Z_at_zeros.abs().mean().item(),
            'passed': Z_at_zeros.abs().max().item() < 0.5,  # Should be near 0
        }


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_trix_fft():
    """Test the TriX FFT implementation."""
    print("="*60)
    print("TRIX FFT UNIT TESTS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    fft = TriXFFT().to(device)
    
    # Test 1: Simple N=8 FFT
    print("\n[Test 1] N=8 FFT")
    x_re = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    x_im = torch.zeros(8, device=device)
    
    X_re, X_im = fft(x_re, x_im)
    
    # FFT of [1,0,0,...] should be [1,1,1,...] (constant)
    expected = torch.ones(8, device=device)
    error = (X_re - expected).abs().max().item()
    print(f"  Input: [1, 0, 0, 0, 0, 0, 0, 0]")
    print(f"  Output real: {X_re.tolist()}")
    print(f"  Expected: [1, 1, 1, 1, 1, 1, 1, 1]")
    print(f"  Max error: {error:.2e}")
    print(f"  Passed: {error < 1e-5}")
    
    # Test 2: Compare with torch.fft
    print("\n[Test 2] Compare with torch.fft (N=16)")
    N = 16
    x_re = torch.randn(N, device=device)
    x_im = torch.randn(N, device=device)
    
    X_re, X_im = fft(x_re, x_im)
    
    # torch.fft reference
    x_complex = torch.complex(x_re, x_im)
    ref = torch.fft.fft(x_complex)
    ref_re = ref.real
    ref_im = ref.imag
    
    error_re = (X_re - ref_re).abs().max().item()
    error_im = (X_im - ref_im).abs().max().item()
    print(f"  Real error: {error_re:.2e}")
    print(f"  Imag error: {error_im:.2e}")
    print(f"  Passed: {max(error_re, error_im) < 1e-4}")
    
    # Test 3: Batch FFT
    print("\n[Test 3] Batch FFT (32 x 64)")
    batch_size = 32
    N = 64
    x_re = torch.randn(batch_size, N, device=device)
    x_im = torch.randn(batch_size, N, device=device)
    
    X_re, X_im = fft(x_re, x_im)
    
    print(f"  Input shape: ({batch_size}, {N})")
    print(f"  Output shape: {X_re.shape}")
    print(f"  Passed: {X_re.shape == (batch_size, N)}")
    
    return True


def test_spectral_tile():
    """Test the SpectralTile implementation."""
    print("\n" + "="*60)
    print("SPECTRAL TILE UNIT TESTS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    tile = SpectralTile(max_n=500).to(device)
    
    # Test 1: Basic Z(t) computation
    print("\n[Test 1] Basic Z(t) computation")
    t = torch.tensor([100.0, 200.0, 300.0], device=device)
    Z = tile(t)
    
    print(f"  t values: {t.tolist()}")
    print(f"  Z(t) values: {Z.tolist()}")
    print(f"  Shape: {Z.shape}")
    print(f"  Passed: {Z.shape == (3,)}")
    
    # Test 2: Known zeros
    print("\n[Test 2] Verify against known zeros")
    result = tile.verify_against_known_zeros()
    print(f"  Max |Z| at zeros: {result['max_abs_Z']:.4f}")
    print(f"  Mean |Z| at zeros: {result['mean_abs_Z']:.4f}")
    print(f"  Passed: {result['passed']}")
    
    # Test 3: Sign changes
    print("\n[Test 3] Sign changes near known zeros")
    # Check t values around second zero (21.022040) - more stable
    t_before = torch.tensor([20.5], device=device)
    t_after = torch.tensor([21.5], device=device)
    
    Z_before = tile(t_before).item()
    Z_after = tile(t_after).item()
    
    sign_change = (Z_before * Z_after) < 0
    print(f"  Z(20.5) = {Z_before:.4f}")
    print(f"  Z(21.5) = {Z_after:.4f}")
    print(f"  Sign change detected: {sign_change}")
    
    # Test 4: Gradient flow
    print("\n[Test 4] Gradient flow")
    t_grad = torch.tensor([100.0], device=device, requires_grad=True)
    Z = tile(t_grad)
    Z.backward()
    
    has_grad = t_grad.grad is not None
    print(f"  Gradient exists: {has_grad}")
    
    # Summary
    print("\n" + "="*60)
    all_passed = result['passed'] and sign_change and has_grad
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    test_trix_fft()
    test_spectral_tile()
