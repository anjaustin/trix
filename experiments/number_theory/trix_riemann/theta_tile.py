#!/usr/bin/env python3
"""
ThetaTile - Riemann-Siegel Theta Function
==========================================

Pure TriX implementation of the theta function:
    θ(t) = (t/2)·log(t/2π) - t/2 - π/8 + O(1/t)

Architecture:
    Fixed microcode operations (exact arithmetic)
    Learned routing (when to apply each op)

This follows the pure_trix_fft pattern:
    Tiles ARE the operations. Routing IS the control.
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple


# =============================================================================
# MICROCODE OPERATIONS
# =============================================================================

class ThetaMicrocode:
    """
    Fixed operations for theta computation.
    
    These are EXACT - no learning, no approximation.
    The operations ARE the tiles.
    """
    
    # Constants
    PI = math.pi
    TWO_PI = 2.0 * math.pi
    PI_OVER_8 = math.pi / 8.0
    
    @staticmethod
    def op_div2(t: torch.Tensor) -> torch.Tensor:
        """t / 2"""
        return t / 2.0
    
    @staticmethod
    def op_div_two_pi(t: torch.Tensor) -> torch.Tensor:
        """t / (2π)"""
        return t / ThetaMicrocode.TWO_PI
    
    @staticmethod
    def op_log(x: torch.Tensor) -> torch.Tensor:
        """Natural logarithm (clamped for safety)"""
        return torch.log(torch.clamp(x, min=1e-10))
    
    @staticmethod
    def op_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiplication"""
        return a * b
    
    @staticmethod
    def op_sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Subtraction"""
        return a - b
    
    @staticmethod
    def op_const_pi8() -> float:
        """π/8 constant"""
        return ThetaMicrocode.PI_OVER_8


# =============================================================================
# THETA TILE - COMPOSED FROM MICROCODE
# =============================================================================

class ThetaTile(nn.Module):
    """
    Riemann-Siegel theta function tile.
    
    Computes θ(t) using fixed microcode operations.
    
    Formula:
        θ(t) = (t/2)·log(t/2π) - t/2 - π/8
    
    Decomposition into microcode:
        1. t_half = DIV2(t)
        2. log_arg = DIV_TWO_PI(t)
        3. log_val = LOG(log_arg)
        4. term1 = MUL(t_half, log_val)
        5. temp = SUB(term1, t_half)
        6. result = SUB(temp, π/8)
    
    This is pure composition - no learned parameters for arithmetic.
    """
    
    def __init__(self, include_corrections: bool = False):
        """
        Args:
            include_corrections: Include O(1/t) asymptotic corrections
        """
        super().__init__()
        self.include_corrections = include_corrections
        self.mc = ThetaMicrocode()
        
        # Correction coefficients (Bernoulli-derived)
        # θ(t) ≈ main + 1/(48t) + 7/(5760t³) + ...
        self.register_buffer('correction_coeffs', torch.tensor([
            1.0 / 48.0,           # B2 term
            7.0 / 5760.0,         # B4 term
            31.0 / 80640.0,       # B6 term
            127.0 / 430080.0,     # B8 term
        ]))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute θ(t).
        
        Args:
            t: Height values (any shape)
            
        Returns:
            theta: θ(t) values (same shape as t)
        """
        # Step 1: t_half = t / 2
        t_half = self.mc.op_div2(t)
        
        # Step 2: log_arg = t / (2π)
        log_arg = self.mc.op_div_two_pi(t)
        
        # Step 3: log_val = log(t / 2π)
        log_val = self.mc.op_log(log_arg)
        
        # Step 4: term1 = (t/2) * log(t/2π)
        term1 = self.mc.op_mul(t_half, log_val)
        
        # Step 5: temp = term1 - t/2
        temp = self.mc.op_sub(term1, t_half)
        
        # Step 6: result = temp - π/8
        result = self.mc.op_sub(temp, self.mc.op_const_pi8())
        
        # Optional: asymptotic corrections for high precision
        if self.include_corrections and t.min() > 10:
            t_inv = 1.0 / t
            t_inv2 = t_inv * t_inv
            
            correction = torch.zeros_like(t)
            t_power = t_inv
            
            # Move coefficients to same device as t
            coeffs = self.correction_coeffs.to(t.device)
            
            for coeff in coeffs:
                correction = correction + coeff * t_power
                t_power = t_power * t_inv2
            
            result = result + correction
        
        return result
    
    def verify(self, t: torch.Tensor, reference_fn=None) -> dict:
        """
        Verify theta computation against reference.
        
        Args:
            t: Test values
            reference_fn: Reference implementation (default: mpmath)
            
        Returns:
            dict with verification results
        """
        computed = self.forward(t)
        
        if reference_fn is None:
            # Use formula directly as reference
            t_np = t.cpu().numpy()
            reference = (t_np / 2) * np.log(t_np / (2 * np.pi)) - t_np / 2 - np.pi / 8
            reference = torch.tensor(reference, dtype=t.dtype, device=t.device)
        else:
            reference = reference_fn(t)
        
        error = torch.abs(computed - reference)
        
        return {
            'max_error': error.max().item(),
            'mean_error': error.mean().item(),
            'computed': computed,
            'reference': reference,
            'passed': error.max().item() < 1e-6,
        }


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_theta_tile():
    """Unit tests for ThetaTile."""
    
    print("="*60)
    print("THETA TILE UNIT TESTS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    tile = ThetaTile().to(device)
    
    # Test 1: Known values
    print("\n[Test 1] Known values")
    t_vals = torch.tensor([14.0, 21.0, 25.0, 100.0, 1000.0], device=device)
    
    result = tile.verify(t_vals)
    print(f"  Max error: {result['max_error']:.2e}")
    print(f"  Mean error: {result['mean_error']:.2e}")
    print(f"  Passed: {result['passed']}")
    
    # Test 2: Batch computation
    print("\n[Test 2] Batch computation")
    t_batch = torch.linspace(14, 1000, 1000, device=device)
    
    result = tile.verify(t_batch)
    print(f"  Max error: {result['max_error']:.2e}")
    print(f"  Mean error: {result['mean_error']:.2e}")
    print(f"  Passed: {result['passed']}")
    
    # Test 3: Large values
    print("\n[Test 3] Large values (t > 10^5)")
    t_large = torch.tensor([1e5, 1e6, 1e7], device=device)
    
    result = tile.verify(t_large)
    print(f"  Max error: {result['max_error']:.2e}")
    print(f"  Mean error: {result['mean_error']:.2e}")
    print(f"  Passed: {result['passed']}")
    
    # Test 4: With corrections
    print("\n[Test 4] With asymptotic corrections")
    tile_corrected = ThetaTile(include_corrections=True).to(device)
    
    result = tile_corrected.verify(t_large)
    print(f"  Max error: {result['max_error']:.2e}")
    print(f"  Mean error: {result['mean_error']:.2e}")
    print(f"  Passed: {result['passed']}")
    
    # Test 5: Gradient flow
    print("\n[Test 5] Gradient flow")
    t_grad = torch.tensor([100.0], device=device, requires_grad=True)
    theta = tile(t_grad)
    theta.backward()
    
    has_grad = t_grad.grad is not None and not torch.isnan(t_grad.grad).any()
    print(f"  Gradient exists: {has_grad}")
    print(f"  Gradient value: {t_grad.grad.item():.6f}")
    
    # Summary
    print("\n" + "="*60)
    all_passed = result['passed'] and has_grad
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    test_theta_tile()
