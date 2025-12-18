#!/usr/bin/env python3
"""
DirichletTile - Dirichlet Series Coefficients
===============================================

Pure TriX implementation of Dirichlet coefficient generation.

Generates coefficients for:
    a_n(t) = n^{-1/2} · e^{-it·log(n)}
           = n^{-1/2} · (cos(t·log(n)) - i·sin(t·log(n)))

Architecture:
    Fixed microcode operations (exact arithmetic)
    Vectorized over n = 1..N
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import torch.nn as nn
import math
from typing import Tuple


# =============================================================================
# MICROCODE OPERATIONS
# =============================================================================

class DirichletMicrocode:
    """
    Fixed operations for Dirichlet coefficient computation.
    
    All operations are EXACT.
    """
    
    @staticmethod
    def op_log(n: torch.Tensor) -> torch.Tensor:
        """Natural logarithm"""
        return torch.log(n)
    
    @staticmethod
    def op_sqrt(n: torch.Tensor) -> torch.Tensor:
        """Square root"""
        return torch.sqrt(n)
    
    @staticmethod
    def op_rsqrt(n: torch.Tensor) -> torch.Tensor:
        """Reciprocal square root: 1/sqrt(n)"""
        return torch.rsqrt(n)
    
    @staticmethod
    def op_cos(x: torch.Tensor) -> torch.Tensor:
        """Cosine"""
        return torch.cos(x)
    
    @staticmethod
    def op_sin(x: torch.Tensor) -> torch.Tensor:
        """Sine"""
        return torch.sin(x)
    
    @staticmethod
    def op_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiplication"""
        return a * b
    
    @staticmethod
    def op_neg(x: torch.Tensor) -> torch.Tensor:
        """Negation"""
        return -x


# =============================================================================
# DIRICHLET TILE
# =============================================================================

class DirichletTile(nn.Module):
    """
    Dirichlet series coefficient generator.
    
    For the Riemann-Siegel formula, we need:
        a_n(t) = n^{-1/2} · e^{-it·log(n)}
    
    In real/imag form:
        real_n(t) = n^{-1/2} · cos(t·log(n))
        imag_n(t) = -n^{-1/2} · sin(t·log(n))
    
    Microcode decomposition:
        1. log_n = LOG(n) for n = 1..N
        2. coeffs = RSQRT(n) for n = 1..N
        3. phases = MUL(t, log_n) - outer product
        4. real = MUL(coeffs, COS(phases))
        5. imag = NEG(MUL(coeffs, SIN(phases)))
    """
    
    def __init__(self, max_n: int = 1000):
        """
        Args:
            max_n: Maximum number of terms to precompute
        """
        super().__init__()
        self.max_n = max_n
        self.mc = DirichletMicrocode()
        
        # Precompute n-dependent values (fixed)
        n_vals = torch.arange(1, max_n + 1, dtype=torch.float32)
        
        # log(n) for n = 1..N
        self.register_buffer('log_n', self.mc.op_log(n_vals))
        
        # n^{-1/2} for n = 1..N
        self.register_buffer('coeffs', self.mc.op_rsqrt(n_vals))
    
    def forward(self, t: torch.Tensor, N: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Dirichlet coefficients for given t values.
        
        Args:
            t: Height values, shape (M,) or (M, 1)
            N: Number of terms (default: auto based on t)
            
        Returns:
            real: Real parts, shape (M, N)
            imag: Imaginary parts, shape (M, N)
        """
        # Determine N based on t if not specified
        if N is None:
            # Riemann-Siegel: N = floor(sqrt(t / 2π))
            t_min = t.min().item()
            N = max(10, min(int(math.sqrt(t_min / (2 * math.pi))), self.max_n))
        
        N = min(N, self.max_n)
        
        # Ensure t is 2D for broadcasting
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (M, 1)
        
        # Get precomputed values for n = 1..N (move to same device as t)
        log_n = self.log_n[:N].to(t.device).unsqueeze(0)    # (1, N)
        coeffs = self.coeffs[:N].to(t.device).unsqueeze(0)  # (1, N)
        
        # Compute phases: t * log(n) for all (t, n) pairs
        # Shape: (M, N)
        phases = self.mc.op_mul(t, log_n)
        
        # Compute real part: n^{-1/2} * cos(t * log(n))
        cos_phases = self.mc.op_cos(phases)
        real = self.mc.op_mul(coeffs, cos_phases)
        
        # Compute imag part: -n^{-1/2} * sin(t * log(n))
        sin_phases = self.mc.op_sin(phases)
        imag = self.mc.op_neg(self.mc.op_mul(coeffs, sin_phases))
        
        return real, imag
    
    def forward_sum(self, t: torch.Tensor, N: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Dirichlet sum directly.
        
        Returns:
            sum_real: Sum of real parts, shape (M,)
            sum_imag: Sum of imaginary parts, shape (M,)
        """
        real, imag = self.forward(t, N)
        return real.sum(dim=1), imag.sum(dim=1)
    
    def verify(self, t: torch.Tensor, N: int = 100) -> dict:
        """
        Verify coefficient computation against direct calculation.
        """
        real, imag = self.forward(t, N)
        
        # Reference computation
        t_np = t.cpu().numpy()
        n_vals = torch.arange(1, N + 1, dtype=torch.float32, device=t.device)
        
        ref_real = torch.zeros(len(t), N, device=t.device)
        ref_imag = torch.zeros(len(t), N, device=t.device)
        
        for i, t_val in enumerate(t):
            for j, n in enumerate(n_vals):
                coeff = 1.0 / torch.sqrt(n)
                phase = t_val * torch.log(n)
                ref_real[i, j] = coeff * torch.cos(phase)
                ref_imag[i, j] = -coeff * torch.sin(phase)
        
        real_error = torch.abs(real - ref_real).max().item()
        imag_error = torch.abs(imag - ref_imag).max().item()
        
        return {
            'real_max_error': real_error,
            'imag_max_error': imag_error,
            'passed': max(real_error, imag_error) < 1e-4,  # float32 tolerance
        }


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_dirichlet_tile():
    """Unit tests for DirichletTile."""
    
    print("="*60)
    print("DIRICHLET TILE UNIT TESTS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    tile = DirichletTile(max_n=500).to(device)
    
    # Test 1: Shape verification
    print("\n[Test 1] Shape verification")
    t = torch.tensor([100.0, 200.0, 300.0], device=device)
    real, imag = tile(t, N=100)
    
    expected_shape = (3, 100)
    shape_ok = real.shape == expected_shape and imag.shape == expected_shape
    print(f"  Expected shape: {expected_shape}")
    print(f"  Real shape: {real.shape}")
    print(f"  Imag shape: {imag.shape}")
    print(f"  Passed: {shape_ok}")
    
    # Test 2: First coefficient (n=1)
    print("\n[Test 2] First coefficient (n=1)")
    # For n=1: log(1)=0, so phase=0, coeff=1
    # real = 1 * cos(0) = 1
    # imag = -1 * sin(0) = 0
    first_real = real[:, 0]
    first_imag = imag[:, 0]
    
    real_ok = torch.allclose(first_real, torch.ones_like(first_real), atol=1e-6)
    imag_ok = torch.allclose(first_imag, torch.zeros_like(first_imag), atol=1e-6)
    print(f"  real[n=1] = {first_real.tolist()}, expected [1, 1, 1]: {real_ok}")
    print(f"  imag[n=1] = {first_imag.tolist()}, expected [0, 0, 0]: {imag_ok}")
    print(f"  Passed: {real_ok and imag_ok}")
    
    # Test 3: Verification against direct computation
    print("\n[Test 3] Verification against direct computation")
    result = tile.verify(t, N=50)
    print(f"  Real max error: {result['real_max_error']:.2e}")
    print(f"  Imag max error: {result['imag_max_error']:.2e}")
    print(f"  Passed: {result['passed']}")
    
    # Test 4: Batch computation
    print("\n[Test 4] Batch computation (1000 t values)")
    t_batch = torch.linspace(100, 10000, 1000, device=device)
    real_batch, imag_batch = tile(t_batch, N=100)
    
    batch_ok = real_batch.shape == (1000, 100)
    print(f"  Shape: {real_batch.shape}")
    print(f"  Passed: {batch_ok}")
    
    # Test 5: Sum computation
    print("\n[Test 5] Sum computation")
    sum_real, sum_imag = tile.forward_sum(t, N=100)
    print(f"  Sum shape: {sum_real.shape}")
    print(f"  Sum real: {sum_real.tolist()}")
    print(f"  Sum imag: {sum_imag.tolist()}")
    sum_ok = sum_real.shape == (3,)
    print(f"  Passed: {sum_ok}")
    
    # Test 6: Gradient flow
    print("\n[Test 6] Gradient flow")
    t_grad = torch.tensor([100.0], device=device, requires_grad=True)
    real_g, imag_g = tile(t_grad, N=50)
    loss = real_g.sum() + imag_g.sum()
    loss.backward()
    
    has_grad = t_grad.grad is not None and not torch.isnan(t_grad.grad).any()
    print(f"  Gradient exists: {has_grad}")
    print(f"  Gradient value: {t_grad.grad.item():.6f}")
    
    # Summary
    print("\n" + "="*60)
    all_passed = shape_ok and real_ok and imag_ok and result['passed'] and batch_ok and sum_ok and has_grad
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    test_dirichlet_tile()
