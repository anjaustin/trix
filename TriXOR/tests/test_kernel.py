"""
TriX Kernel Tests

Verifies correctness and performance of the sparse 2-bit kernel.
"""

import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch

from trix.kernel import (
    pack_weights,
    unpack_weights,
    trix_forward,
    TriXLinear,
    is_neon_available,
)


class TestPackUnpack:
    """Tests for weight packing and unpacking."""
    
    def test_roundtrip(self):
        """Pack and unpack should return original weights."""
        torch.manual_seed(42)
        
        rows, cols = 64, 128
        weights = torch.sign(torch.randn(rows, cols))
        
        packed = pack_weights(weights)
        unpacked = unpack_weights(packed, rows, cols)
        
        assert torch.allclose(weights, unpacked), "Pack/unpack roundtrip failed"
    
    def test_encoding_values(self):
        """Verify specific encoding values."""
        weights = torch.tensor([[1, -1, 0, 1], [0, 0, -1, -1]], dtype=torch.float32)
        packed = pack_weights(weights)
        
        byte0 = packed[0, 0].item()
        assert (byte0 >> 0) & 0x03 == 0x01, "+1 should encode as 0x01"
        assert (byte0 >> 2) & 0x03 == 0x02, "-1 should encode as 0x02"
        assert (byte0 >> 4) & 0x03 == 0x00, "0 should encode as 0x00"
        assert (byte0 >> 6) & 0x03 == 0x01, "+1 should encode as 0x01"
    
    def test_compression_ratio(self):
        """Packed weights should be 4x smaller in element count."""
        rows, cols = 64, 256
        weights = torch.sign(torch.randn(rows, cols))
        packed = pack_weights(weights)
        
        expected_packed_cols = cols // 4
        actual_packed_cols = packed.shape[1]
        
        assert actual_packed_cols == expected_packed_cols


class TestForward:
    """Tests for forward pass computation."""
    
    def test_basic_forward(self):
        """Forward pass should match reference matmul."""
        torch.manual_seed(42)
        
        batch, in_f, out_f, num_tiles = 2, 8, 8, 2
        
        input_data = torch.randn(batch, in_f)
        weights = torch.sign(torch.randn(out_f, in_f))
        scales = torch.ones(out_f)
        gate_mask = torch.ones(batch, num_tiles)
        
        packed = pack_weights(weights)
        output = trix_forward(input_data, packed, scales, gate_mask, out_f, num_tiles)
        
        expected = input_data @ weights.T
        
        assert torch.allclose(output, expected, atol=1e-5)
    
    def test_tile_gating(self):
        """Inactive tiles should produce zero output."""
        torch.manual_seed(42)
        
        batch, in_f, out_f, num_tiles = 2, 16, 16, 4
        out_per_tile = out_f // num_tiles
        
        input_data = torch.randn(batch, in_f)
        weights = torch.sign(torch.randn(out_f, in_f))
        scales = torch.ones(out_f)
        
        gate_mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
        
        packed = pack_weights(weights)
        output = trix_forward(input_data, packed, scales, gate_mask, out_f, num_tiles)
        
        for b in range(batch):
            for t in range(num_tiles):
                start = t * out_per_tile
                end = start + out_per_tile
                if gate_mask[b, t] == 0:
                    assert torch.allclose(output[b, start:end], torch.zeros(out_per_tile))


class TestSpeedup:
    """Performance tests for sparse computation."""
    
    @pytest.mark.skipif(not is_neon_available(), reason="C++ library not available")
    def test_speedup_at_75_percent_sparsity(self):
        """Verify speedup when 75% of tiles are inactive."""
        torch.manual_seed(42)
        batch, in_f, out_f, num_tiles = 32, 512, 2048, 4
        
        input_data = torch.randn(batch, in_f)
        weights = torch.sign(torch.randn(out_f, in_f))
        scales = torch.ones(out_f)
        packed = pack_weights(weights)
        
        gate_all = torch.ones(batch, num_tiles)
        gate_quarter = torch.zeros(batch, num_tiles)
        gate_quarter[:, 0] = 1
        
        # Warmup
        for _ in range(3):
            trix_forward(input_data, packed, scales, gate_all, out_f, num_tiles)
        
        n_iters = 20
        
        start = time.perf_counter()
        for _ in range(n_iters):
            trix_forward(input_data, packed, scales, gate_all, out_f, num_tiles)
        time_all = (time.perf_counter() - start) / n_iters
        
        start = time.perf_counter()
        for _ in range(n_iters):
            trix_forward(input_data, packed, scales, gate_quarter, out_f, num_tiles)
        time_quarter = (time.perf_counter() - start) / n_iters
        
        speedup = time_all / time_quarter
        
        assert speedup >= 2.0, f"Expected at least 2x speedup, got {speedup:.2f}x"
        print(f"\nSpeedup at 75% sparsity: {speedup:.2f}x")


class TestPyTorchIntegration:
    """Tests for PyTorch module integration."""
    
    def test_module_forward(self):
        """TriXLinear should produce valid output."""
        torch.manual_seed(42)
        
        batch, in_f, out_f, num_tiles = 4, 64, 128, 4
        
        layer = TriXLinear(in_f, out_f, num_tiles)
        x = torch.randn(batch, in_f)
        gate = torch.ones(batch, num_tiles)
        
        layer.train()
        out = layer(x, gate)
        
        assert out.shape == (batch, out_f)
    
    def test_gradient_flow(self):
        """Gradients should flow through the layer."""
        torch.manual_seed(42)
        
        batch, in_f, out_f, num_tiles = 4, 64, 128, 4
        
        layer = TriXLinear(in_f, out_f, num_tiles)
        x = torch.randn(batch, in_f)
        gate = torch.ones(batch, num_tiles)
        
        layer.train()
        out = layer(x, gate)
        loss = out.sum()
        loss.backward()
        
        assert layer.scales.grad is not None
        assert not torch.all(layer.scales.grad == 0)
    
    @pytest.mark.skipif(not is_neon_available(), reason="C++ library not available")
    def test_pytorch_cpp_match(self):
        """PyTorch and C++ implementations should match."""
        torch.manual_seed(42)
        
        batch, in_f, out_f, num_tiles = 4, 64, 128, 4
        
        layer = TriXLinear(in_f, out_f, num_tiles)
        x = torch.randn(batch, in_f)
        gate = torch.ones(batch, num_tiles)
        
        layer.train()
        out_pytorch = layer(x, gate)
        
        layer.eval()
        layer.pack()
        out_cpp = layer(x, gate)
        
        max_diff = (out_pytorch - out_cpp).abs().max().item()
        assert max_diff < 1e-4


class TestNumericalAccuracy:
    """Tests for numerical precision."""
    
    def test_accumulation_accuracy(self):
        """Large matmuls should maintain accuracy."""
        torch.manual_seed(42)
        
        batch, in_f, out_f, num_tiles = 8, 1024, 4096, 4
        
        input_data = torch.randn(batch, in_f)
        weights = torch.sign(torch.randn(out_f, in_f))
        scales = torch.ones(out_f)
        gate_mask = torch.ones(batch, num_tiles)
        
        packed = pack_weights(weights)
        output = trix_forward(input_data, packed, scales, gate_mask, out_f, num_tiles)
        
        expected = input_data @ weights.T
        
        max_diff = (output - expected).abs().max().item()
        assert max_diff < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
