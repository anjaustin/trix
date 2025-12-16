#!/usr/bin/env python3
"""
Rigorous Test Suite for TriX Compiler and Transforms

This test suite is designed for researchers who want to validate our claims.
Every test is:
- Deterministic (seeded randomness)
- Exhaustive where feasible
- Documented with what it proves

Run with: pytest tests/test_rigorous.py -v
"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'experiments/fft_atoms')

import pytest
import numpy as np
import torch
from itertools import product


# =============================================================================
# SECTION 1: FP4 ATOM VERIFICATION
# =============================================================================

class TestFP4Atoms:
    """
    Verify that FP4 threshold circuit atoms compute exactly.
    
    Claim: Each atom computes its boolean function perfectly on all inputs.
    Method: Exhaustive truth table verification.
    """
    
    def test_and_atom_exhaustive(self):
        """AND gate: output 1 iff both inputs are 1."""
        from trix.compiler.atoms_fp4 import FP4AtomLibrary
        
        lib = FP4AtomLibrary()
        atom = lib.get_atom("AND")
        
        truth_table = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 1,
        }
        
        for inputs, expected in truth_table.items():
            x = torch.tensor([list(inputs)], dtype=torch.float32)
            y = atom(x)  # Atoms are callable directly
            actual = int(y[0, 0].item() > 0.5)
            assert actual == expected, f"AND({inputs}) = {actual}, expected {expected}"
    
    def test_or_atom_exhaustive(self):
        """OR gate: output 1 iff at least one input is 1."""
        from trix.compiler.atoms_fp4 import FP4AtomLibrary
        
        lib = FP4AtomLibrary()
        atom = lib.get_atom("OR")
        
        truth_table = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 1,
        }
        
        for inputs, expected in truth_table.items():
            x = torch.tensor([list(inputs)], dtype=torch.float32)
            y = atom(x)
            actual = int(y[0, 0].item() > 0.5)
            assert actual == expected, f"OR({inputs}) = {actual}, expected {expected}"
    
    def test_xor_atom_exhaustive(self):
        """XOR gate: output 1 iff inputs differ."""
        from trix.compiler.atoms_fp4 import FP4AtomLibrary
        
        lib = FP4AtomLibrary()
        atom = lib.get_atom("XOR")
        
        truth_table = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 0,
        }
        
        for inputs, expected in truth_table.items():
            x = torch.tensor([list(inputs)], dtype=torch.float32)
            y = atom(x)
            actual = int(y[0, 0].item() > 0.5)
            assert actual == expected, f"XOR({inputs}) = {actual}, expected {expected}"
    
    def test_sum_atom_exhaustive(self):
        """SUM (half-adder sum): output is XOR of 3 inputs."""
        from trix.compiler.atoms_fp4 import FP4AtomLibrary
        
        lib = FP4AtomLibrary()
        atom = lib.get_atom("SUM")
        
        # SUM(a, b, c) = a XOR b XOR c = parity
        for a, b, c in product([0, 1], repeat=3):
            expected = (a + b + c) % 2
            x = torch.tensor([[float(a), float(b), float(c)]], dtype=torch.float32)
            y = atom(x)
            actual = int(y[0, 0].item() > 0.5)
            assert actual == expected, f"SUM({a},{b},{c}) = {actual}, expected {expected}"
    
    def test_carry_atom_exhaustive(self):
        """CARRY (half-adder carry): output 1 iff at least 2 inputs are 1."""
        from trix.compiler.atoms_fp4 import FP4AtomLibrary
        
        lib = FP4AtomLibrary()
        atom = lib.get_atom("CARRY")
        
        # CARRY(a, b, c) = majority = 1 iff sum >= 2
        for a, b, c in product([0, 1], repeat=3):
            expected = 1 if (a + b + c) >= 2 else 0
            x = torch.tensor([[float(a), float(b), float(c)]], dtype=torch.float32)
            y = atom(x)
            actual = int(y[0, 0].item() > 0.5)
            assert actual == expected, f"CARRY({a},{b},{c}) = {actual}, expected {expected}"
    
    def test_all_atoms_registered(self):
        """Verify all 10 claimed atoms exist and are retrievable."""
        from trix.compiler.atoms_fp4 import FP4AtomLibrary
        
        lib = FP4AtomLibrary()
        expected_atoms = ["AND", "OR", "XOR", "NOT", "NAND", "NOR", "XNOR", "SUM", "CARRY", "MUX"]
        
        for name in expected_atoms:
            atom = lib.get_atom(name)
            assert atom is not None, f"Atom {name} not found"
            assert atom.status.name == "VERIFIED", f"Atom {name} not verified"


# =============================================================================
# SECTION 2: ADDER VERIFICATION
# =============================================================================

class TestAdderExhaustive:
    """
    Verify that compiled adders compute exactly.
    
    Claim: 8-bit adder produces correct sum for all inputs.
    Method: Exhaustive verification (2^16 = 65,536 cases).
    """
    
    def test_full_adder_exhaustive(self):
        """
        Full adder: sum and carry for all 8 input combinations.
        
        Inputs: a, b, cin (3 bits)
        Outputs: sum, cout (2 bits)
        """
        from trix.compiler.atoms_fp4 import FP4AtomLibrary
        
        lib = FP4AtomLibrary()
        sum_atom = lib.get_atom("SUM")
        carry_atom = lib.get_atom("CARRY")
        
        for a, b, cin in product([0, 1], repeat=3):
            # Expected
            total = a + b + cin
            expected_sum = total % 2
            expected_cout = total // 2
            
            # Actual
            x = torch.tensor([[float(a), float(b), float(cin)]], dtype=torch.float32)
            actual_sum = int(sum_atom(x)[0, 0].item() > 0.5)
            actual_cout = int(carry_atom(x)[0, 0].item() > 0.5)
            
            assert actual_sum == expected_sum, f"SUM({a},{b},{cin}) = {actual_sum}, expected {expected_sum}"
            assert actual_cout == expected_cout, f"CARRY({a},{b},{cin}) = {actual_cout}, expected {expected_cout}"
    
    @pytest.mark.slow
    def test_8bit_adder_exhaustive(self):
        """
        8-bit adder: exhaustive verification of all 65,536 input pairs.
        
        This test takes ~30 seconds. Mark with @pytest.mark.slow.
        Run with: pytest -v -m slow
        """
        from trix.compiler.atoms_fp4 import FP4AtomLibrary
        
        lib = FP4AtomLibrary()
        sum_atom = lib.get_atom("SUM")
        carry_atom = lib.get_atom("CARRY")
        
        def add_8bit(a: int, b: int) -> int:
            """Simulate 8-bit adder using SUM and CARRY atoms."""
            a_bits = [(a >> i) & 1 for i in range(8)]
            b_bits = [(b >> i) & 1 for i in range(8)]
            
            carry = 0
            result_bits = []
            
            for i in range(8):
                x = torch.tensor([[float(a_bits[i]), float(b_bits[i]), float(carry)]], dtype=torch.float32)
                s = int(sum_atom(x)[0, 0].item() > 0.5)
                c = int(carry_atom(x)[0, 0].item() > 0.5)
                result_bits.append(s)
                carry = c
            
            result = sum(b << i for i, b in enumerate(result_bits))
            return result
        
        errors = []
        for a in range(256):
            for b in range(256):
                expected = (a + b) % 256  # 8-bit wrap
                actual = add_8bit(a, b)
                if actual != expected:
                    errors.append((a, b, expected, actual))
        
        assert len(errors) == 0, f"8-bit adder failed on {len(errors)} cases. First 5: {errors[:5]}"


# =============================================================================
# SECTION 3: TRANSFORM VERIFICATION
# =============================================================================

class TestWHTExhaustive:
    """
    Verify Walsh-Hadamard Transform.
    
    Claim: Compiled WHT matches scipy.linalg.hadamard exactly.
    Method: Test against scipy reference for multiple N.
    """
    
    def test_wht_n8_against_scipy(self):
        """WHT N=8 matches scipy Hadamard matrix multiplication."""
        from scipy.linalg import hadamard
        from fft_compiler import compile_fft_routing, CompiledWHT
        
        N = 8
        routing = compile_fft_routing(N)
        wht = CompiledWHT(
            N=N,
            is_upper_circuit=routing['is_upper']['circuit'],
            partner_circuits=routing['partner']['circuits'],
        )
        
        H = hadamard(N)
        
        # Test 100 random integer vectors
        np.random.seed(42)  # Reproducible
        for _ in range(100):
            x = np.random.randint(0, 10, size=N).astype(float)
            
            expected = H @ x
            actual = np.array(wht.execute(list(x)))
            
            np.testing.assert_array_equal(actual, expected)
    
    def test_wht_self_inverse(self):
        """WHT is self-inverse: WHT(WHT(x)) = N * x."""
        from fft_compiler import compile_fft_routing, CompiledWHT
        
        for N in [8, 16]:
            routing = compile_fft_routing(N)
            wht = CompiledWHT(
                N=N,
                is_upper_circuit=routing['is_upper']['circuit'],
                partner_circuits=routing['partner']['circuits'],
            )
            
            np.random.seed(42)
            for _ in range(20):
                x = list(np.random.randint(0, 10, size=N).astype(float))
                
                wht_x = wht.execute(x)
                wht_wht_x = wht.execute(wht_x)
                
                # WHT(WHT(x)) = N * x
                recovered = [v / N for v in wht_wht_x]
                
                np.testing.assert_array_almost_equal(recovered, x)


class TestDFTExhaustive:
    """
    Verify Discrete Fourier Transform with twiddle opcodes.
    
    Claim: Compiled DFT matches numpy.fft.fft to float precision.
    Method: Test against numpy reference, verify no runtime trig.
    """
    
    def test_dft_n8_against_numpy(self):
        """DFT N=8 matches numpy.fft.fft."""
        from fft_compiler import compile_complex_fft_routing, CompiledComplexFFT
        
        N = 8
        routing = compile_complex_fft_routing(N)
        dft = CompiledComplexFFT(
            N=N,
            is_upper_circuit=routing['is_upper']['circuit'],
            partner_circuits=routing['partner']['circuits'],
            twiddle_circuits=routing['twiddle']['circuits'],
        )
        
        np.random.seed(42)
        for _ in range(50):
            x = np.random.randn(N)
            
            expected = np.fft.fft(x)
            out_re, out_im = dft.execute(list(x))
            actual = np.array(out_re) + 1j * np.array(out_im)
            
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    
    def test_dft_no_runtime_trig(self):
        """Verify execute() contains no runtime trig calls."""
        from fft_compiler import verify_no_runtime_trig
        
        # This will raise AssertionError if trig is found
        assert verify_no_runtime_trig() == True
    
    def test_dft_twiddle_coverage(self):
        """Verify correct twiddle opcodes are used for N=8."""
        from fft_compiler import get_twiddle_index
        
        N = 8
        twiddle_used = set()
        
        for stage in range(3):  # log2(8) = 3
            m = 2 ** (stage + 1)
            half_m = m // 2
            for j in range(half_m):
                tw_idx = get_twiddle_index(N, m, j)
                twiddle_used.add(tw_idx)
        
        # DIT FFT uses N/2 unique twiddles
        assert twiddle_used == {0, 1, 2, 3}, f"Expected {{0,1,2,3}}, got {twiddle_used}"
    
    def test_dft_inverse_roundtrip(self):
        """Verify DFT -> IDFT roundtrip recovers original."""
        from fft_compiler import compile_complex_fft_routing, CompiledComplexFFT
        
        N = 8
        routing = compile_complex_fft_routing(N)
        dft = CompiledComplexFFT(
            N=N,
            is_upper_circuit=routing['is_upper']['circuit'],
            partner_circuits=routing['partner']['circuits'],
            twiddle_circuits=routing['twiddle']['circuits'],
        )
        
        np.random.seed(42)
        for _ in range(20):
            x = np.random.randn(N)
            
            # Forward
            out_re, out_im = dft.execute(list(x))
            X = np.array(out_re) + 1j * np.array(out_im)
            
            # Inverse via numpy (validates our forward is correct)
            recovered = np.fft.ifft(X).real
            
            np.testing.assert_allclose(recovered, x, rtol=1e-10, atol=1e-10)


# =============================================================================
# SECTION 4: COMPOSITION VERIFICATION
# =============================================================================

class TestAtomComposition:
    """
    Verify that atoms compose correctly when wired together.
    
    Claim: Chaining atoms preserves correctness.
    Method: Test specific compositions against known results.
    """
    
    def test_xor_chain(self):
        """XOR(XOR(a, b), c) = a XOR b XOR c."""
        from trix.compiler.atoms_fp4 import FP4AtomLibrary
        
        lib = FP4AtomLibrary()
        xor_atom = lib.get_atom("XOR")
        
        for a, b, c in product([0, 1], repeat=3):
            # First XOR
            x1 = torch.tensor([[float(a), float(b)]], dtype=torch.float32)
            ab = float(xor_atom(x1)[0, 0].item() > 0.5)
            
            # Second XOR
            x2 = torch.tensor([[ab, float(c)]], dtype=torch.float32)
            actual = int(xor_atom(x2)[0, 0].item() > 0.5)
            
            expected = a ^ b ^ c
            assert actual == expected, f"XOR(XOR({a},{b}),{c}) = {actual}, expected {expected}"
    
    def test_and_or_composition(self):
        """(a AND b) OR (c AND d) computes correctly."""
        from trix.compiler.atoms_fp4 import FP4AtomLibrary
        
        lib = FP4AtomLibrary()
        and_atom = lib.get_atom("AND")
        or_atom = lib.get_atom("OR")
        
        for a, b, c, d in product([0, 1], repeat=4):
            # a AND b
            x1 = torch.tensor([[float(a), float(b)]], dtype=torch.float32)
            ab = float(and_atom(x1)[0, 0].item() > 0.5)
            
            # c AND d
            x2 = torch.tensor([[float(c), float(d)]], dtype=torch.float32)
            cd = float(and_atom(x2)[0, 0].item() > 0.5)
            
            # OR
            x3 = torch.tensor([[ab, cd]], dtype=torch.float32)
            actual = int(or_atom(x3)[0, 0].item() > 0.5)
            
            expected = (a & b) | (c & d)
            assert actual == expected


# =============================================================================
# SECTION 5: EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """
    Test behavior on edge cases and adversarial inputs.
    """
    
    def test_wht_all_zeros(self):
        """WHT of all zeros is all zeros."""
        from fft_compiler import compile_fft_routing, CompiledWHT
        
        N = 8
        routing = compile_fft_routing(N)
        wht = CompiledWHT(
            N=N,
            is_upper_circuit=routing['is_upper']['circuit'],
            partner_circuits=routing['partner']['circuits'],
        )
        
        x = [0.0] * N
        result = wht.execute(x)
        
        assert all(v == 0 for v in result)
    
    def test_wht_impulse(self):
        """WHT of impulse [1,0,0,...] is all ones."""
        from fft_compiler import compile_fft_routing, CompiledWHT
        
        N = 8
        routing = compile_fft_routing(N)
        wht = CompiledWHT(
            N=N,
            is_upper_circuit=routing['is_upper']['circuit'],
            partner_circuits=routing['partner']['circuits'],
        )
        
        x = [1.0] + [0.0] * (N - 1)
        result = wht.execute(x)
        
        assert all(v == 1 for v in result)
    
    def test_dft_dc_component(self):
        """DFT of constant signal has energy only at DC."""
        from fft_compiler import compile_complex_fft_routing, CompiledComplexFFT
        
        N = 8
        routing = compile_complex_fft_routing(N)
        dft = CompiledComplexFFT(
            N=N,
            is_upper_circuit=routing['is_upper']['circuit'],
            partner_circuits=routing['partner']['circuits'],
            twiddle_circuits=routing['twiddle']['circuits'],
        )
        
        x = [1.0] * N
        out_re, out_im = dft.execute(x)
        
        # DC component should be N
        assert abs(out_re[0] - N) < 1e-10
        assert abs(out_im[0]) < 1e-10
        
        # All other components should be 0
        for i in range(1, N):
            assert abs(out_re[i]) < 1e-10
            assert abs(out_im[i]) < 1e-10


# =============================================================================
# SECTION 6: NUMERICAL STABILITY
# =============================================================================

class TestNumericalStability:
    """
    Test numerical behavior with extreme values.
    """
    
    def test_dft_large_values(self):
        """DFT handles large input values."""
        from fft_compiler import compile_complex_fft_routing, CompiledComplexFFT
        
        N = 8
        routing = compile_complex_fft_routing(N)
        dft = CompiledComplexFFT(
            N=N,
            is_upper_circuit=routing['is_upper']['circuit'],
            partner_circuits=routing['partner']['circuits'],
            twiddle_circuits=routing['twiddle']['circuits'],
        )
        
        x = [1e6] * N
        out_re, out_im = dft.execute(x)
        expected = np.fft.fft(x)
        
        np.testing.assert_allclose(
            np.array(out_re) + 1j * np.array(out_im),
            expected,
            rtol=1e-9
        )
    
    def test_dft_small_values(self):
        """DFT handles small input values."""
        from fft_compiler import compile_complex_fft_routing, CompiledComplexFFT
        
        N = 8
        routing = compile_complex_fft_routing(N)
        dft = CompiledComplexFFT(
            N=N,
            is_upper_circuit=routing['is_upper']['circuit'],
            partner_circuits=routing['partner']['circuits'],
            twiddle_circuits=routing['twiddle']['circuits'],
        )
        
        x = [1e-10] * N
        out_re, out_im = dft.execute(x)
        expected = np.fft.fft(x)
        
        np.testing.assert_allclose(
            np.array(out_re) + 1j * np.array(out_im),
            expected,
            rtol=1e-5  # Looser tolerance for small values
        )


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
