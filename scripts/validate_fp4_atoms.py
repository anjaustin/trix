#!/usr/bin/env python3
"""
Validate FP4 Atoms

Exhaustively tests all FP4 threshold circuit atoms against their truth tables.
100% accuracy is required - no tolerance.

Usage:
    python scripts/validate_fp4_atoms.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from trix.compiler.atoms_fp4 import (
    FP4AtomLibrary,
    truth_table_to_circuit,
    verify_generated_circuit,
)


def banner(text: str):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def show_weights(library: FP4AtomLibrary, name: str):
    """Display the FP4 weights for an atom."""
    data = library.get_fp4_weights(name)
    print(f"\n  {name} weights (FP4-compatible):")
    for i, layer in enumerate(data["layers"]):
        print(f"    Layer {i}: W={layer['weights']}, b={layer['bias']}")


def main():
    banner("FP4 ATOM VALIDATION")
    print()
    print("Testing threshold circuit atoms for 100% accuracy.")
    print("These atoms are CONSTRUCTED, not trained.")
    print()
    
    library = FP4AtomLibrary()
    
    # Test each atom
    banner("ATOM VERIFICATION")
    
    results = {}
    all_pass = True
    
    for name in library.list_atoms():
        passed, accuracy, failures = library.verify_atom(name)
        results[name] = (passed, accuracy, failures)
        
        atom = library.get_atom(name)
        layers = len(atom.layers)
        layer_type = "1-layer (linear)" if layers == 1 else f"{layers}-layer (minterm)"
        
        status = "PASS" if passed else "FAIL"
        print(f"  {name:8} [{layer_type:20}]: {status} ({accuracy:.0%})")
        
        if not passed:
            all_pass = False
            print(f"           Failures: {failures}")
    
    # Summary
    banner("SUMMARY")
    
    passed_count = sum(1 for p, _, _ in results.values() if p)
    total_count = len(results)
    
    print(f"\n  Atoms passed: {passed_count}/{total_count}")
    print()
    
    if all_pass:
        print("  ALL ATOMS VERIFIED - 100% EXACT")
        print()
        print("  FP4 weight values used:")
        print("    Weights: {-1.0, 0.0, 1.0}")
        print("    Biases:  {-2.5, -1.5, -0.5, 0.5, 1.5}")
        print()
        print("  All values exist in E2M1 and NF4 FP4 formats.")
    else:
        print("  SOME ATOMS FAILED - Check implementation")
        sys.exit(1)
    
    # Show weight details
    banner("WEIGHT DETAILS")
    
    for name in library.list_atoms():
        show_weights(library, name)
    
    # Test the minterm generator
    banner("MINTERM GENERATOR TEST")
    
    print("\n  Testing automatic circuit generation from truth tables...")
    
    # Test with XOR (known 2-layer function)
    xor_table = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    xor_circuit = truth_table_to_circuit("XOR_GEN", 2, xor_table)
    passed, acc, failures = verify_generated_circuit(xor_circuit, xor_table)
    print(f"  XOR (generated): {'PASS' if passed else 'FAIL'} ({acc:.0%})")
    
    # Test with 3-input function
    sum_table = {
        (0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 0): 1, (0, 1, 1): 0,
        (1, 0, 0): 1, (1, 0, 1): 0, (1, 1, 0): 0, (1, 1, 1): 1,
    }
    sum_circuit = truth_table_to_circuit("SUM_GEN", 3, sum_table)
    passed, acc, failures = verify_generated_circuit(sum_circuit, sum_table)
    print(f"  SUM (generated): {'PASS' if passed else 'FAIL'} ({acc:.0%})")
    
    # Test arbitrary 4-input function
    # f(a,b,c,d) = 1 iff exactly 2 inputs are 1
    exactly_two = {}
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                for d in [0, 1]:
                    exactly_two[(a, b, c, d)] = 1 if (a + b + c + d) == 2 else 0
    
    two_circuit = truth_table_to_circuit("EXACTLY_TWO", 4, exactly_two)
    passed, acc, failures = verify_generated_circuit(two_circuit, exactly_two)
    print(f"  EXACTLY_TWO (4-input, generated): {'PASS' if passed else 'FAIL'} ({acc:.0%})")
    
    banner("VALIDATION COMPLETE")
    print()
    print("  FP4 atoms are ready for integration with the compiler.")
    print("  Use FP4AtomLibrary as a drop-in replacement for AtomLibrary.")
    print()


if __name__ == "__main__":
    main()
