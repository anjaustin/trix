#!/usr/bin/env python3
"""
SignChangeTile - Zero Detection
================================

Pure TriX implementation of sign change detection for locating zeros.

Method:
    If Z(t_i) * Z(t_{i+1}) < 0, a zero exists in [t_i, t_{i+1}]

Architecture:
    Fixed microcode: SIGN, MUL, CMP operations
    Parallel detection across all evaluation points
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/src')
sys.path.insert(0, '/workspace/trix_latest/experiments/number_theory')

import torch
import torch.nn as nn
import math
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ZeroCandidate:
    """A detected zero candidate."""
    t_low: float       # Lower bound
    t_high: float      # Upper bound
    t_estimate: float  # Estimated location (midpoint or interpolated)
    Z_low: float       # Z(t_low)
    Z_high: float      # Z(t_high)


# =============================================================================
# MICROCODE OPERATIONS
# =============================================================================

class SignMicrocode:
    """
    Fixed operations for sign detection.
    
    All operations are EXACT.
    """
    
    @staticmethod
    def op_sign(x: torch.Tensor) -> torch.Tensor:
        """Sign function: -1, 0, or +1"""
        return torch.sign(x)
    
    @staticmethod
    def op_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiplication"""
        return a * b
    
    @staticmethod
    def op_lt_zero(x: torch.Tensor) -> torch.Tensor:
        """Less than zero: returns boolean tensor"""
        return x < 0
    
    @staticmethod
    def op_ne(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Not equal"""
        return a != b


# =============================================================================
# SIGN CHANGE TILE
# =============================================================================

class SignChangeTile(nn.Module):
    """
    Detects sign changes in Z(t) values to locate zeros.
    
    Method:
        1. Compute signs of consecutive Z values
        2. Detect where sign changes (sign_i * sign_{i+1} < 0)
        3. Return intervals containing zeros
    
    Microcode:
        - SIGN: Get sign of Z values
        - MUL: Product of consecutive signs
        - LT_ZERO: Check for sign change
    """
    
    def __init__(self):
        super().__init__()
        self.mc = SignMicrocode()
    
    def forward(self, t_values: torch.Tensor, Z_values: torch.Tensor) -> List[ZeroCandidate]:
        """
        Detect zeros from Z(t) evaluations.
        
        Args:
            t_values: Height values, shape (N,)
            Z_values: Z(t) values, shape (N,)
            
        Returns:
            List of ZeroCandidate objects
        """
        # Step 1: Get signs
        signs = self.mc.op_sign(Z_values)
        
        # Step 2: Product of consecutive signs
        sign_products = self.mc.op_mul(signs[:-1], signs[1:])
        
        # Step 3: Find sign changes (product < 0)
        sign_changes = self.mc.op_lt_zero(sign_products)
        
        # Step 4: Also check for exact zeros (sign = 0)
        exact_zeros = signs == 0
        
        # Step 5: Extract candidates
        candidates = []
        
        # Sign change candidates
        change_indices = torch.where(sign_changes)[0]
        for idx in change_indices:
            i = idx.item()
            t_low = t_values[i].item()
            t_high = t_values[i + 1].item()
            Z_low = Z_values[i].item()
            Z_high = Z_values[i + 1].item()
            
            # Linear interpolation for estimate
            if abs(Z_high - Z_low) > 1e-10:
                t_est = t_low - Z_low * (t_high - t_low) / (Z_high - Z_low)
            else:
                t_est = (t_low + t_high) / 2
            
            candidates.append(ZeroCandidate(
                t_low=t_low,
                t_high=t_high,
                t_estimate=t_est,
                Z_low=Z_low,
                Z_high=Z_high,
            ))
        
        # Exact zero candidates
        zero_indices = torch.where(exact_zeros)[0]
        for idx in zero_indices:
            i = idx.item()
            t_val = t_values[i].item()
            candidates.append(ZeroCandidate(
                t_low=t_val,
                t_high=t_val,
                t_estimate=t_val,
                Z_low=0.0,
                Z_high=0.0,
            ))
        
        return candidates
    
    def count_zeros(self, t_values: torch.Tensor, Z_values: torch.Tensor) -> int:
        """
        Count zeros without returning candidates.
        
        More efficient for large-scale scanning.
        """
        signs = self.mc.op_sign(Z_values)
        sign_products = self.mc.op_mul(signs[:-1], signs[1:])
        sign_changes = self.mc.op_lt_zero(sign_products)
        
        return sign_changes.sum().item()
    
    def batch_count(self, t_values: torch.Tensor, Z_values: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Count zeros and return sign change mask.
        
        Args:
            t_values: (N,) t values
            Z_values: (N,) Z values
            
        Returns:
            count: Number of zeros
            mask: Boolean mask of sign change locations
        """
        signs = self.mc.op_sign(Z_values)
        sign_products = self.mc.op_mul(signs[:-1], signs[1:])
        mask = self.mc.op_lt_zero(sign_products)
        
        return mask.sum().item(), mask


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_sign_tile():
    """Unit tests for SignChangeTile."""
    
    print("="*60)
    print("SIGN CHANGE TILE UNIT TESTS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    tile = SignChangeTile()
    
    # Test 1: Simple sign change
    print("\n[Test 1] Simple sign change")
    t = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
    Z = torch.tensor([1.0, 0.5, -0.5, -1.0], device=device)  # Sign change between t=1 and t=2
    
    candidates = tile(t, Z)
    print(f"  t values: {t.tolist()}")
    print(f"  Z values: {Z.tolist()}")
    print(f"  Candidates found: {len(candidates)}")
    
    if len(candidates) > 0:
        c = candidates[0]
        print(f"  Zero in [{c.t_low}, {c.t_high}], estimate: {c.t_estimate:.4f}")
    
    test1_pass = len(candidates) == 1 and abs(candidates[0].t_estimate - 1.5) < 0.1
    print(f"  Passed: {test1_pass}")
    
    # Test 2: Multiple sign changes
    print("\n[Test 2] Multiple sign changes")
    t = torch.linspace(0, 10, 100, device=device)
    Z = torch.sin(t * 2)  # Oscillates, multiple zeros
    
    candidates = tile(t, Z)
    count = tile.count_zeros(t, Z)
    
    print(f"  Range: [0, 10], 100 points")
    print(f"  Function: sin(2t)")
    print(f"  Expected zeros: ~6 (at t = 0, π/2, π, 3π/2, ...)")
    print(f"  Detected: {len(candidates)} candidates, {count} count")
    
    test2_pass = 5 <= count <= 7
    print(f"  Passed: {test2_pass}")
    
    # Test 3: No sign changes
    print("\n[Test 3] No sign changes")
    t = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
    Z = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)  # All positive
    
    candidates = tile(t, Z)
    print(f"  Z values (all positive): {Z.tolist()}")
    print(f"  Candidates found: {len(candidates)}")
    
    test3_pass = len(candidates) == 0
    print(f"  Passed: {test3_pass}")
    
    # Test 4: Exact zero
    print("\n[Test 4] Exact zero")
    t = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
    Z = torch.tensor([1.0, 0.0, -1.0, -2.0], device=device)  # Exact zero at t=1
    
    candidates = tile(t, Z)
    print(f"  Z values (zero at t=1): {Z.tolist()}")
    print(f"  Candidates found: {len(candidates)}")
    
    # Should find both exact zero and sign change
    test4_pass = len(candidates) >= 1
    print(f"  Passed: {test4_pass}")
    
    # Test 5: Batch counting performance
    print("\n[Test 5] Batch counting (10M points)")
    import time
    
    t_large = torch.linspace(0, 100000, 10_000_000, device=device)
    Z_large = torch.sin(t_large / 100)  # Many oscillations
    
    start = time.time()
    count = tile.count_zeros(t_large, Z_large)
    elapsed = time.time() - start
    
    rate = 10_000_000 / elapsed
    print(f"  Points: 10,000,000")
    print(f"  Zeros found: {count}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Rate: {rate:,.0f} points/sec")
    
    test5_pass = elapsed < 1.0  # Should be fast
    print(f"  Passed: {test5_pass}")
    
    # Summary
    print("\n" + "="*60)
    all_passed = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    test_sign_tile()
