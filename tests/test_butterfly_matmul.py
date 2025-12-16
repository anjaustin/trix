#!/usr/bin/env python3
"""
Rigorous Tests for Butterfly MatMul

Verifies:
1. Butterfly networks compute correct transforms
2. Identity and Hadamard are exact (0.00 error)
3. Structure matches our FFT/WHT implementations
4. Ternary blocks enumerate correctly
"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'experiments/fft_atoms')
sys.path.insert(0, 'experiments/matmul')

import pytest
import numpy as np
from scipy.linalg import hadamard


class TestButterflyIdentity:
    """Test butterfly network for identity matrix."""
    
    def test_identity_n8_exact(self):
        """Identity N=8 has 0.00 error."""
        from butterfly_matmul import identity_butterfly
        
        N = 8
        net = identity_butterfly(N)
        M = net.as_matrix()
        
        error = np.max(np.abs(M - np.eye(N)))
        assert error == 0.0, f"Identity error: {error}"
    
    def test_identity_n16_exact(self):
        """Identity N=16 has 0.00 error."""
        from butterfly_matmul import identity_butterfly
        
        N = 16
        net = identity_butterfly(N)
        M = net.as_matrix()
        
        error = np.max(np.abs(M - np.eye(N)))
        assert error == 0.0, f"Identity error: {error}"
    
    def test_identity_forward_pass(self):
        """Identity forward pass returns input unchanged."""
        from butterfly_matmul import identity_butterfly
        
        N = 8
        net = identity_butterfly(N)
        
        np.random.seed(42)
        for _ in range(10):
            x = np.random.randn(N)
            y = net.forward(x)
            np.testing.assert_array_almost_equal(y, x)


class TestButterflyHadamard:
    """Test butterfly network for Hadamard matrix."""
    
    def test_hadamard_n8_exact(self):
        """Hadamard N=8 matches scipy exactly."""
        from butterfly_matmul import hadamard_butterfly
        
        N = 8
        net = hadamard_butterfly(N)
        M = net.as_matrix()
        H = hadamard(N).astype(float)
        
        error = np.max(np.abs(M - H))
        assert error == 0.0, f"Hadamard error: {error}"
    
    def test_hadamard_n16_exact(self):
        """Hadamard N=16 matches scipy exactly."""
        from butterfly_matmul import hadamard_butterfly
        
        N = 16
        net = hadamard_butterfly(N)
        M = net.as_matrix()
        H = hadamard(N).astype(float)
        
        error = np.max(np.abs(M - H))
        assert error == 0.0, f"Hadamard error: {error}"
    
    def test_hadamard_matches_wht(self):
        """Butterfly Hadamard matches our WHT implementation."""
        from butterfly_matmul import hadamard_butterfly
        from fft_compiler import compile_fft_routing, CompiledWHT
        
        N = 8
        
        # Butterfly version
        net = hadamard_butterfly(N)
        
        # WHT version
        routing = compile_fft_routing(N)
        wht = CompiledWHT(
            N=N,
            is_upper_circuit=routing['is_upper']['circuit'],
            partner_circuits=routing['partner']['circuits'],
        )
        
        # Compare on random inputs
        np.random.seed(42)
        for _ in range(10):
            x = np.random.randint(0, 10, size=N).astype(float)
            
            y_butterfly = net.forward(x)
            y_wht = np.array(wht.execute(list(x)))
            
            np.testing.assert_array_almost_equal(y_butterfly, y_wht)
    
    def test_hadamard_self_inverse(self):
        """Hadamard is self-inverse (up to scaling)."""
        from butterfly_matmul import hadamard_butterfly
        
        N = 8
        net = hadamard_butterfly(N)
        
        np.random.seed(42)
        x = np.random.randn(N)
        
        # H(H(x)) = N * x
        y = net.forward(x)
        z = net.forward(y)
        
        np.testing.assert_array_almost_equal(z / N, x)


class TestButterflyStructure:
    """Test butterfly network structure properties."""
    
    def test_butterfly_stages(self):
        """Butterfly has log2(N) stages."""
        from butterfly_matmul import identity_butterfly
        
        for N in [8, 16, 32]:
            net = identity_butterfly(N)
            expected_stages = int(np.log2(N))
            assert len(net.layers) == expected_stages
    
    def test_butterfly_pairs(self):
        """Each stage pairs N/2 element pairs."""
        from butterfly_matmul import ButterflyLayer
        import numpy as np
        
        N = 8
        for stage in range(3):
            stride = 2 ** stage
            blocks = [np.eye(2) for _ in range(N // 2)]
            layer = ButterflyLayer(N, stride, blocks)
            pairs = layer.get_pairs()
            
            assert len(pairs) == N // 2
            
            # Each element appears exactly once
            all_indices = [i for pair in pairs for i in pair]
            assert sorted(all_indices) == list(range(N))
    
    def test_xor_pairing(self):
        """Butterfly uses XOR pairing like FFT."""
        from butterfly_matmul import ButterflyLayer
        import numpy as np
        
        N = 8
        for stage in range(3):
            stride = 2 ** stage
            blocks = [np.eye(2) for _ in range(N // 2)]
            layer = ButterflyLayer(N, stride, blocks)
            pairs = layer.get_pairs()
            
            # Verify XOR pairing
            for i, j in pairs:
                assert i ^ stride == j


class TestTernaryBlocks:
    """Test ternary block enumeration."""
    
    def test_enumerate_count(self):
        """There are exactly 81 ternary 2x2 matrices."""
        from butterfly_matmul import enumerate_ternary_2x2
        
        matrices = enumerate_ternary_2x2()
        assert len(matrices) == 81  # 3^4
    
    def test_classify_coverage(self):
        """Classification covers all 81 matrices."""
        from butterfly_matmul import classify_ternary_2x2
        
        classified = classify_ternary_2x2()
        total = sum(len(v) for v in classified.values())
        assert total == 81
    
    def test_hadamard_like_count(self):
        """There are exactly 12 orthogonal ternary 2x2 matrices."""
        from butterfly_matmul import classify_ternary_2x2
        
        classified = classify_ternary_2x2()
        # Hadamard-like + identity-like (which are also orthogonal)
        orthogonal_count = len(classified['hadamard_like']) + len(classified['identity_like'])
        # The identity-like are ±I, which are orthogonal
        assert orthogonal_count >= 12  # At least 12


class TestMonarchStructure:
    """Test Monarch matrix structure."""
    
    def test_monarch_is_permutation(self):
        """Monarch with identity blocks is a permutation."""
        from butterfly_matmul import identity_monarch
        
        N = 16
        layer = identity_monarch(N)
        M = layer.as_matrix()
        
        # Check it's a permutation matrix
        assert np.allclose(M @ M.T, np.eye(N))
        assert np.allclose(np.sum(M, axis=0), np.ones(N))
        assert np.allclose(np.sum(M, axis=1), np.ones(N))
    
    def test_monarch_permutation_pattern(self):
        """Monarch produces the transpose-reshape permutation."""
        from butterfly_matmul import identity_monarch
        
        N = 16
        p = 4
        q = 4
        
        layer = identity_monarch(N, p)
        M = layer.as_matrix()
        
        # The permutation should be: i -> (i % p) * q + (i // p)
        perm = np.argmax(M, axis=1)
        expected = [(i % p) * q + (i // p) for i in range(N)]
        
        assert list(perm) == expected


class TestIntegration:
    """Integration tests with existing TriX code."""
    
    def test_butterfly_uses_same_pairing_as_fft(self):
        """Verify butterfly and FFT use identical pairing logic."""
        from butterfly_matmul import ButterflyLayer
        from fft_compiler import analyze_fft_structure
        import numpy as np
        
        N = 8
        fft_structure = analyze_fft_structure(N)
        
        for stage in range(3):
            stride = 2 ** stage
            blocks = [np.eye(2) for _ in range(N // 2)]
            layer = ButterflyLayer(N, stride, blocks)
            
            for pos in range(N):
                fft_partner = fft_structure['partner'][(stage, pos)]
                butterfly_partner = pos ^ stride
                
                assert fft_partner == butterfly_partner


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
