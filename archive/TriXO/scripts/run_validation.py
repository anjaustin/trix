#!/usr/bin/env python3
"""
TriX Validation Runner

Simple script to validate the entire TriX system.
Run this to verify your installation and the correctness of all components.

Usage:
    python scripts/run_validation.py          # Quick validation
    python scripts/run_validation.py --full   # Full validation (includes slow tests)
"""

import sys
import argparse

sys.path.insert(0, 'src')
sys.path.insert(0, 'experiments/fft_atoms')


def validate_atoms():
    """Validate FP4 atoms."""
    print("\n" + "=" * 60)
    print("VALIDATING FP4 ATOMS")
    print("=" * 60)
    
    from trix.compiler.atoms_fp4 import FP4AtomLibrary
    import torch
    
    lib = FP4AtomLibrary()
    atoms_to_test = ["AND", "OR", "XOR", "NOT", "NAND", "NOR", "XNOR", "SUM", "CARRY", "MUX"]
    
    all_passed = True
    for name in atoms_to_test:
        atom = lib.get_atom(name)
        if atom is None:
            print(f"  {name}: NOT FOUND")
            all_passed = False
        elif atom.status.name != "VERIFIED":
            print(f"  {name}: {atom.status.name}")
            all_passed = False
        else:
            print(f"  {name}: VERIFIED (100%)")
    
    return all_passed


def validate_adder():
    """Validate full adder composition."""
    print("\n" + "=" * 60)
    print("VALIDATING FULL ADDER")
    print("=" * 60)
    
    from trix.compiler.atoms_fp4 import FP4AtomLibrary
    import torch
    from itertools import product
    
    lib = FP4AtomLibrary()
    sum_atom = lib.get_atom("SUM")
    carry_atom = lib.get_atom("CARRY")
    
    errors = 0
    for a, b, cin in product([0, 1], repeat=3):
        x = torch.tensor([[float(a), float(b), float(cin)]], dtype=torch.float32)
        
        actual_sum = int(sum_atom(x)[0, 0].item() > 0.5)
        actual_cout = int(carry_atom(x)[0, 0].item() > 0.5)
        
        total = a + b + cin
        expected_sum = total % 2
        expected_cout = total // 2
        
        if actual_sum != expected_sum or actual_cout != expected_cout:
            errors += 1
    
    if errors == 0:
        print(f"  Full Adder: 8/8 cases PASSED (100%)")
        return True
    else:
        print(f"  Full Adder: {8-errors}/8 cases passed")
        return False


def validate_wht():
    """Validate Walsh-Hadamard Transform."""
    print("\n" + "=" * 60)
    print("VALIDATING WALSH-HADAMARD TRANSFORM")
    print("=" * 60)
    
    from fft_compiler import compile_fft_routing, CompiledWHT
    from scipy.linalg import hadamard
    import numpy as np
    
    for N in [8, 16]:
        routing = compile_fft_routing(N)
        wht = CompiledWHT(
            N=N,
            is_upper_circuit=routing['is_upper']['circuit'],
            partner_circuits=routing['partner']['circuits'],
        )
        
        H = hadamard(N)
        
        # Test 10 random vectors
        np.random.seed(42)
        errors = 0
        for _ in range(10):
            x = np.random.randint(0, 10, size=N).astype(float)
            expected = H @ x
            actual = np.array(wht.execute(list(x)))
            if not np.allclose(actual, expected):
                errors += 1
        
        if errors == 0:
            print(f"  WHT N={N}: PASSED (10/10 tests)")
        else:
            print(f"  WHT N={N}: FAILED ({10-errors}/10 tests)")
            return False
    
    return True


def validate_dft():
    """Validate Discrete Fourier Transform with twiddle opcodes."""
    print("\n" + "=" * 60)
    print("VALIDATING DFT (TWIDDLE OPCODES)")
    print("=" * 60)
    
    from fft_compiler import compile_complex_fft_routing, CompiledComplexFFT, verify_no_runtime_trig
    import numpy as np
    
    # Check no runtime trig
    try:
        verify_no_runtime_trig()
        print("  No runtime trig: PASSED")
    except AssertionError as e:
        print(f"  No runtime trig: FAILED - {e}")
        return False
    
    for N in [8, 16]:
        routing = compile_complex_fft_routing(N)
        dft = CompiledComplexFFT(
            N=N,
            is_upper_circuit=routing['is_upper']['circuit'],
            partner_circuits=routing['partner']['circuits'],
            twiddle_circuits=routing['twiddle']['circuits'],
        )
        
        np.random.seed(42)
        max_error = 0
        for _ in range(10):
            x = np.random.randn(N)
            expected = np.fft.fft(x)
            out_re, out_im = dft.execute(list(x))
            actual = np.array(out_re) + 1j * np.array(out_im)
            error = np.max(np.abs(actual - expected))
            max_error = max(max_error, error)
        
        if max_error < 1e-10:
            print(f"  DFT N={N}: PASSED (max error: {max_error:.2e})")
        else:
            print(f"  DFT N={N}: FAILED (max error: {max_error:.2e})")
            return False
    
    return True


def run_full_validation():
    """Run pytest on the rigorous test suite."""
    print("\n" + "=" * 60)
    print("RUNNING FULL TEST SUITE (pytest)")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/test_rigorous.py", "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="TriX Validation Runner")
    parser.add_argument("--full", action="store_true", help="Run full test suite with pytest")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRIX VALIDATION")
    print("=" * 60)
    
    results = {}
    
    # Quick validations
    results["atoms"] = validate_atoms()
    results["adder"] = validate_adder()
    results["wht"] = validate_wht()
    results["dft"] = validate_dft()
    
    # Full validation if requested
    if args.full:
        results["pytest"] = run_full_validation()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
