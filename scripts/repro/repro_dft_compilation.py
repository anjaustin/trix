#!/usr/bin/env python3
"""
Repro script: DFT Compilation

Claim: Compiled DFT matches numpy.fft.fft to float precision (0.00 error at N=8).

This script compiles a DFT using the TriX fft_compiler (twiddle opcodes,
no runtime trig), runs it on random inputs, and compares to numpy.

Expected output: see repro_dft_compilation.expected.json

Usage:
    python scripts/repro/repro_dft_compilation.py

Note: Requires experiments/fft_atoms to be present (repo checkout, not wheel).
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments" / "fft_atoms"))

try:
    from fft_compiler import (
        compile_complex_fft_routing,
        CompiledComplexFFT,
        verify_no_runtime_trig,
    )
except ImportError:
    print("SKIP: requires experiments/fft_atoms (not available in wheel installs)")
    sys.exit(0)


def main():
    np.random.seed(42)

    N = 8
    routing = compile_complex_fft_routing(N)
    dft = CompiledComplexFFT(
        N=N,
        is_upper_circuit=routing["is_upper"]["circuit"],
        partner_circuits=routing["partner"]["circuits"],
        twiddle_circuits=routing["twiddle"]["circuits"],
    )

    # Verify no runtime trig
    no_trig = verify_no_runtime_trig()

    # Run on 100 random inputs and measure max error
    max_error = 0.0
    for _ in range(100):
        x = np.random.randn(N)
        expected = np.fft.fft(x)

        out_re, out_im = dft.execute(list(x))
        actual = np.array(out_re) + 1j * np.array(out_im)

        error = np.max(np.abs(actual - expected))
        max_error = max(max_error, error)

    # Roundtrip: DFT -> IDFT
    x_rt = np.random.randn(N)
    out_re, out_im = dft.execute(list(x_rt))
    X = np.array(out_re) + 1j * np.array(out_im)
    recovered = np.fft.ifft(X).real
    roundtrip_error = np.max(np.abs(recovered - x_rt))

    results = {
        "N": N,
        "max_error": float(f"{max_error:.2e}"),
        "roundtrip_error": float(f"{roundtrip_error:.2e}"),
        "no_runtime_trig": no_trig,
        "num_test_vectors": 100,
    }

    print(json.dumps(results, indent=2))

    # Compare against expected
    expected_path = Path(__file__).with_suffix(".expected.json")
    if expected_path.exists():
        exp = json.loads(expected_path.read_text())
        ok = True
        if results["max_error"] > exp["max_error_tolerance"]:
            print(
                f"FAIL: max_error = {results['max_error']}, expected <= {exp['max_error_tolerance']}"
            )
            ok = False
        if results["no_runtime_trig"] != exp["no_runtime_trig"]:
            print(f"FAIL: no_runtime_trig = {results['no_runtime_trig']}")
            ok = False
        if ok:
            print("PASS")
        sys.exit(0 if ok else 1)
    else:
        print(f"(no expected output file at {expected_path}; skipping comparison)")


if __name__ == "__main__":
    main()
