#!/usr/bin/env python3
"""
TriX 6502: ATOMIC - Simplified

The composition rule for ADC is trivial:
  ADC(A, B, C) = INC(ADD(A, B)) if C else ADD(A, B)

No router needed. Just exact atoms + direct composition.
"""

import torch
import torch.nn as nn


def ADD(a, b):
    """A + B mod 256"""
    return (a + b) % 256

def INC(a):
    """A + 1 mod 256"""
    return (a + 1) % 256

def ADC(a, b, c):
    """A + B + C via composition: ADD then conditionally INC"""
    result = ADD(a, b)
    # If carry, compose with INC
    result = torch.where(c == 1, INC(result), result)
    return result


def verify():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("ATOMIC ADC via COMPOSITION")
    print("=" * 60)
    print("ADC(A, B, C) = INC(ADD(A, B)) if C else ADD(A, B)")
    print()
    
    # Exhaustive test
    errors = 0
    total = 0
    
    for c in [0, 1]:
        for a_val in range(256):
            a = torch.full((256,), a_val, device=device)
            b = torch.arange(256, device=device)
            carry = torch.full((256,), c, device=device)
            
            # Ground truth
            expected = (a + b + carry) % 256
            
            # Atomic composition
            result = ADC(a, b, carry)
            
            errors += (result != expected).sum().item()
            total += 256
    
    acc = (total - errors) / total * 100
    
    print(f"Tested: {total:,} cases (256 × 256 × 2)")
    print(f"Errors: {errors}")
    print(f"Accuracy: {acc:.1f}%")
    
    print()
    print("=" * 60)
    if errors == 0:
        print("100% - ATOMS + COMPOSITION = PERFECT")
    print("=" * 60)
    
    # Show the composition
    print()
    print("The insight:")
    print("  ADD is an atom (A + B)")
    print("  INC is an atom (A + 1)")
    print("  ADC = ADD → INC (conditional composition)")
    print()
    print("Routing IS composition.")
    print("Tiles ARE atoms.")
    print("SpatioTemporal routes based on state (carry).")


if __name__ == "__main__":
    verify()
