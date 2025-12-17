"""
Tests for Mesa 9 (Euler Probe) and Mesa 10 (Chudnovsky Cartridge)
"""

import pytest
import torch
import numpy as np


class TestEulerProbe:
    """Tests for Mesa 9: Euler Probe spectral analysis."""
    
    def test_digit_stream_import(self):
        """Test DigitStream can be imported."""
        from experiments.number_theory.euler_probe import DigitStream
        stream = DigitStream(source='pi')
        assert stream is not None
    
    def test_digit_stream_pi_first_20(self):
        """Test first 20 digits of pi are correct."""
        from experiments.number_theory.euler_probe import DigitStream
        stream = DigitStream(source='pi')
        digits = stream.get_all(20)
        expected = [1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6]
        assert digits == expected
    
    def test_digit_stream_e_first_20(self):
        """Test first 20 digits of e are correct."""
        from experiments.number_theory.euler_probe import DigitStream
        stream = DigitStream(source='e')
        digits = stream.get_all(20)
        expected = [7,1,8,2,8,1,8,2,8,4,5,9,0,4,5,2,3,5,3,6]
        assert digits == expected
    
    def test_spectral_analyzer_import(self):
        """Test SpectralAnalyzer can be imported."""
        from experiments.number_theory.euler_probe import SpectralAnalyzer
        analyzer = SpectralAnalyzer(window_size=64)
        assert analyzer is not None
    
    def test_spectral_analyzer_fft_accuracy(self):
        """Test FFT accuracy vs NumPy."""
        from experiments.number_theory.euler_probe import SpectralAnalyzer
        
        analyzer = SpectralAnalyzer(window_size=64)
        
        # Test signal
        x = torch.randn(64)
        
        # Our FFT (returns tuple of real, imag)
        our_real, our_imag = analyzer.fft(x)
        our_fft = torch.complex(our_real, our_imag)
        
        # NumPy FFT
        np_fft = torch.from_numpy(np.fft.fft(x.numpy()))
        
        # Compare
        error = torch.abs(our_fft - np_fft).max().item()
        assert error < 1e-6, f"FFT error too high: {error}"
    
    def test_accumulator_tile_block_sums(self):
        """Test AccumulatorTile block sum computation."""
        from experiments.number_theory.euler_probe import AccumulatorTile
        
        tile = AccumulatorTile(block_size=10)
        digits = list(range(10)) * 10  # 0-9 repeated 10 times
        
        sums = tile.block_sums(digits)
        
        # Each block of 10 digits (0-9) sums to 45
        expected = [45] * 10
        assert sums == expected


class TestChudnovskyGMP:
    """Tests for Mesa 10: GMP-accelerated Chudnovsky."""
    
    def test_binary_splitting_import(self):
        """Test BinarySplittingChudnovsky can be imported."""
        from experiments.number_theory.chudnovsky_gmp import BinarySplittingChudnovsky
        bs = BinarySplittingChudnovsky(100)
        assert bs is not None
    
    def test_binary_splitting_first_50_digits(self):
        """Test first 50 digits of pi are correct."""
        from experiments.number_theory.chudnovsky_gmp import BinarySplittingChudnovsky
        
        bs = BinarySplittingChudnovsky(100)
        digits = bs.compute_mpfr()
        
        expected = "14159265358979323846264338327950288419716939937510"
        assert digits[:50] == expected
    
    def test_binary_splitting_1000_digits(self):
        """Test 1000 digits computation."""
        from experiments.number_theory.chudnovsky_gmp import BinarySplittingChudnovsky
        
        bs = BinarySplittingChudnovsky(1000)
        digits = bs.compute_mpfr()
        
        # Verify length
        assert len(digits) >= 1000
        
        # Verify first 50
        expected = "14159265358979323846264338327950288419716939937510"
        assert digits[:50] == expected
    
    def test_gmp_closed_loop_import(self):
        """Test GMPClosedLoop can be imported."""
        from experiments.number_theory.chudnovsky_gmp import GMPClosedLoop
        loop = GMPClosedLoop()
        assert loop is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gmp_closed_loop_analyze(self):
        """Test GPU analysis in closed loop."""
        from experiments.number_theory.chudnovsky_gmp import GMPClosedLoop
        
        loop = GMPClosedLoop(window_size=64, block_size=10)
        
        # Random digits
        digits = [i % 10 for i in range(10000)]
        result = loop.analyze_on_gpu(digits)
        
        assert 'whiteness_mean' in result
        assert 'whiteness_std' in result
        assert 'n_windows' in result
        assert result['whiteness_mean'] > 0


class TestCUDABigInt:
    """Tests for CUDA BigInt operations."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_bigint_from_int(self):
        """Test CUDABigInt creation from int."""
        from experiments.number_theory.cuda_bigint import CUDABigInt
        
        n = 12345678901234567890
        bigint = CUDABigInt(n, device='cuda')
        
        assert bigint.to_int() == n
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")  
    def test_cuda_bigint_add(self):
        """Test CUDABigInt addition."""
        from experiments.number_theory.cuda_bigint import CUDABigInt, cuda_bigint_add
        
        a = CUDABigInt(123456789, device='cuda')
        b = CUDABigInt(987654321, device='cuda')
        
        c = cuda_bigint_add(a, b)
        
        assert c.to_int() == 123456789 + 987654321
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_factorial_small(self):
        """Test CUDA factorial for small numbers."""
        from experiments.number_theory.cuda_bigint import cuda_factorial
        
        # Test 10!
        fact_10 = cuda_factorial(10, device='cuda')
        assert fact_10.to_int() == 3628800
        
        # Test 20!
        fact_20 = cuda_factorial(20, device='cuda')
        assert fact_20.to_int() == 2432902008176640000


class TestParallelChudnovsky:
    """Tests for parallel Chudnovsky."""
    
    def test_parallel_import(self):
        """Test ParallelChudnovsky can be imported."""
        from experiments.number_theory.parallel_chudnovsky import ParallelChudnovsky
        pc = ParallelChudnovsky(100, num_workers=1)
        assert pc is not None
    
    def test_parallel_sequential_correctness(self):
        """Test sequential mode produces correct digits."""
        from experiments.number_theory.parallel_chudnovsky import ParallelChudnovsky
        
        pc = ParallelChudnovsky(100, num_workers=1)
        digits = pc.compute()
        
        expected = "14159265358979323846264338327950288419716939937510"
        assert digits[:50] == expected
    
    def test_parallel_multicore_correctness(self):
        """Test parallel mode produces correct digits."""
        from experiments.number_theory.parallel_chudnovsky import ParallelChudnovsky
        
        pc = ParallelChudnovsky(100, num_workers=2)
        digits = pc.compute()
        
        expected = "14159265358979323846264338327950288419716939937510"
        assert digits[:50] == expected


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_closed_loop_small(self):
        """Test full closed loop with small input."""
        from experiments.number_theory.chudnovsky_gmp import GMPClosedLoop
        
        loop = GMPClosedLoop(window_size=64, block_size=10)
        result = loop.run(10000, compare_random=False)
        
        assert result['total_digits'] == 10000
        assert result['gen_rate'] > 0
        assert result['whiteness'] > 0
    
    def test_euler_probe_spectral_whiteness(self):
        """Test spectral whiteness calculation."""
        from experiments.number_theory.euler_probe import SpectralWhitenessTest, DigitStream
        
        stream = DigitStream(source='pi')
        digits = stream.get_all(1000)
        
        test = SpectralWhitenessTest(window_size=64, block_size=10)
        result = test.test_source(digits, "pi")
        
        # Result is a WhitenessResult dataclass
        assert hasattr(result, 'whiteness_score')
        assert result.whiteness_score > 0
