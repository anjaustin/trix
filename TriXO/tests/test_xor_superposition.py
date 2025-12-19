"""
Tests for XOR Superposition Signature Compression

Tests cover:
- SparseDelta creation and properties
- CompressedSignatures compression/decompression roundtrip
- SuperpositionRouter routing equivalence
- XORSuperpositionFFN forward pass
- Batch operations performance
- Compression ratio validation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestSparseDelta:
    """Tests for SparseDelta dataclass."""

    def test_creation(self):
        """Test SparseDelta creation."""
        from trix.nn.xor_superposition import SparseDelta

        positions = torch.tensor([0, 5, 10], dtype=torch.int16)
        values = torch.tensor([1, -1, 1], dtype=torch.int8)
        delta = SparseDelta(positions=positions, values=values)

        assert delta.sparsity == 3
        assert delta.memory_bytes == 3 * 2 + 3 * 1  # positions + values

    def test_to_device(self):
        """Test moving SparseDelta to device."""
        from trix.nn.xor_superposition import SparseDelta

        positions = torch.tensor([0, 1], dtype=torch.int16)
        values = torch.tensor([1, -1], dtype=torch.int8)
        delta = SparseDelta(positions=positions, values=values)

        delta_moved = delta.to(torch.device('cpu'))
        assert delta_moved.positions.device.type == 'cpu'


class TestPackUnpack:
    """Tests for ternary packing/unpacking."""

    def test_pack_unpack_roundtrip(self):
        """Test pack then unpack returns original."""
        from trix.nn.xor_superposition import pack_ternary_to_uint8, unpack_uint8_to_ternary

        # Create ternary tensor
        ternary = torch.tensor([[1, -1, 0, 1, -1, 0, 1, 1]], dtype=torch.float32)
        dim = ternary.shape[-1]

        # Pack and unpack
        packed = pack_ternary_to_uint8(ternary)
        unpacked = unpack_uint8_to_ternary(packed, dim)

        assert torch.allclose(ternary, unpacked)

    def test_pack_compression(self):
        """Test packing achieves 4x compression (element count)."""
        from trix.nn.xor_superposition import pack_ternary_to_uint8

        dim = 512
        ternary = torch.sign(torch.randn(1, dim))

        packed = pack_ternary_to_uint8(ternary)

        assert packed.shape[-1] == (dim + 3) // 4

    def test_pack_batch(self):
        """Test batch packing."""
        from trix.nn.xor_superposition import pack_ternary_to_uint8, unpack_uint8_to_ternary

        batch, dim = 32, 128
        ternary = torch.sign(torch.randn(batch, dim))

        packed = pack_ternary_to_uint8(ternary)
        unpacked = unpack_uint8_to_ternary(packed, dim)

        assert torch.allclose(ternary, unpacked)


class TestCompressedSignatures:
    """Tests for CompressedSignatures class."""

    def test_compression_roundtrip(self):
        """Test compress then decompress returns original."""
        from trix.nn.xor_superposition import CompressedSignatures

        num_tiles = 16
        d_model = 128
        signatures = torch.sign(torch.randn(num_tiles, d_model))

        compressed = CompressedSignatures().compress(signatures)
        decompressed = compressed.decompress_all()

        assert torch.equal(signatures, decompressed)

    def test_compression_single_index(self):
        """Test decompressing single signature."""
        from trix.nn.xor_superposition import CompressedSignatures

        num_tiles = 8
        d_model = 64
        signatures = torch.sign(torch.randn(num_tiles, d_model))

        compressed = CompressedSignatures().compress(signatures)

        for i in range(num_tiles):
            decompressed = compressed.decompress(i)
            assert torch.equal(signatures[i], decompressed)

    def test_compression_ratio_high_similarity(self):
        """Test compression ratio with highly similar signatures."""
        from trix.nn.xor_superposition import CompressedSignatures

        num_tiles = 64
        d_model = 512

        # Create base
        base = torch.sign(torch.randn(d_model))

        # Create variations with ~1% difference
        signatures = []
        for i in range(num_tiles):
            sig = base.clone()
            flip_mask = torch.rand(d_model) < 0.01
            sig[flip_mask] *= -1
            signatures.append(sig)
        signatures = torch.stack(signatures)

        compressed = CompressedSignatures().compress(signatures)
        stats = compressed.get_compression_stats()

        # Should achieve >10x compression with 1% variation
        assert stats.compression_ratio > 10.0
        assert stats.mean_delta_sparsity < 0.02  # <2% differences

    def test_compression_ratio_random(self):
        """Test compression ratio with random signatures."""
        from trix.nn.xor_superposition import CompressedSignatures

        num_tiles = 16
        d_model = 128
        signatures = torch.sign(torch.randn(num_tiles, d_model))

        compressed = CompressedSignatures().compress(signatures)
        stats = compressed.get_compression_stats()

        # Random signatures should still achieve some compression
        # due to centroid-based base
        assert stats.compression_ratio > 1.0

    def test_to_device(self):
        """Test moving compressed signatures to device."""
        from trix.nn.xor_superposition import CompressedSignatures

        signatures = torch.sign(torch.randn(8, 64))
        compressed = CompressedSignatures().compress(signatures)

        compressed_moved = compressed.to(torch.device('cpu'))
        assert compressed_moved.base_ternary.device.type == 'cpu'


class TestSuperpositionRouter:
    """Tests for SuperpositionRouter."""

    def test_routing_basic(self):
        """Test basic routing functionality."""
        from trix.nn.xor_superposition import SuperpositionRouter

        num_tiles = 8
        d_model = 64
        batch = 16

        router = SuperpositionRouter(num_tiles, d_model)
        x = torch.randn(batch, d_model)

        tile_idx, scores = router.route(x, return_scores=True)

        assert tile_idx.shape == (batch,)
        assert scores.shape == (batch, num_tiles)
        assert (tile_idx >= 0).all() and (tile_idx < num_tiles).all()

    def test_routing_equivalence(self):
        """Test that compressed routing matches uncompressed."""
        from trix.nn.xor_superposition import SuperpositionRouter

        num_tiles = 16
        d_model = 128
        batch = 64

        router = SuperpositionRouter(num_tiles, d_model)
        x = torch.randn(batch, d_model)

        # Route uncompressed
        idx_uncompressed, _ = router.route(x)

        # Compress and route
        router.compress()
        idx_compressed, _ = router.route(x)

        # Should match exactly
        assert router.verify_routing_equivalence(x)

    def test_routing_determinism(self):
        """Test that routing is deterministic."""
        from trix.nn.xor_superposition import SuperpositionRouter

        num_tiles = 8
        d_model = 64

        router = SuperpositionRouter(num_tiles, d_model)
        router.compress()

        x = torch.randn(32, d_model)

        # Multiple calls should give same result
        idx1, _ = router.route(x)
        idx2, _ = router.route(x)
        idx3, _ = router.route(x)

        assert torch.equal(idx1, idx2)
        assert torch.equal(idx2, idx3)

    def test_compression_stats(self):
        """Test compression statistics retrieval."""
        from trix.nn.xor_superposition import SuperpositionRouter

        router = SuperpositionRouter(16, 128)

        # Before compression
        assert router.get_compression_stats() is None

        # After compression
        router.compress()
        stats = router.get_compression_stats()

        assert stats is not None
        assert stats.num_signatures == 16
        assert stats.compression_ratio > 0

    def test_set_signatures(self):
        """Test setting signatures from external source."""
        from trix.nn.xor_superposition import SuperpositionRouter

        router = SuperpositionRouter(8, 64)
        new_sigs = torch.sign(torch.randn(8, 64))

        router.set_signatures(new_sigs)

        assert torch.allclose(router.get_ternary_signatures(), new_sigs)


class TestXORSuperpositionFFN:
    """Tests for XORSuperpositionFFN."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        from trix.nn.xor_superposition import XORSuperpositionFFN

        d_model = 64
        num_tiles = 8
        batch, seq = 4, 16

        ffn = XORSuperpositionFFN(d_model=d_model, num_tiles=num_tiles)
        x = torch.randn(batch, seq, d_model)

        out, routing_info = ffn(x, return_routing_info=True)

        assert out.shape == x.shape
        assert routing_info is not None
        assert 'tile_idx' in routing_info
        assert 'entropy' in routing_info

    def test_forward_2d(self):
        """Test forward pass with 2D input."""
        from trix.nn.xor_superposition import XORSuperpositionFFN

        d_model = 64
        batch = 32

        ffn = XORSuperpositionFFN(d_model=d_model)
        x = torch.randn(batch, d_model)

        out, _ = ffn(x)

        assert out.shape == x.shape

    def test_compress_decompress_cycle(self):
        """Test compression and decompression cycle."""
        from trix.nn.xor_superposition import XORSuperpositionFFN

        ffn = XORSuperpositionFFN(d_model=64, num_tiles=8)
        x = torch.randn(16, 64)

        # Uncompressed forward
        out1, _ = ffn(x)

        # Compress
        ffn.compress()
        stats = ffn.get_compression_stats()
        assert stats is not None

        # Decompress
        ffn.decompress()
        assert ffn.get_compression_stats() is None

    def test_factory_function(self):
        """Test create_compressed_ffn factory."""
        from trix.nn.xor_superposition import create_compressed_ffn

        ffn = create_compressed_ffn(d_model=128, num_tiles=16)
        assert isinstance(ffn, nn.Module)

        x = torch.randn(8, 128)
        out, _ = ffn(x)
        assert out.shape == x.shape


class TestBatchOperations:
    """Tests for batch operations in xor_routing.py."""

    def test_popcount_vectorized(self):
        """Test vectorized popcount."""
        from trix.nn.xor_routing import popcount_vectorized

        # Test known values
        x = torch.tensor([0, 1, 255, 170], dtype=torch.uint8)
        counts = popcount_vectorized(x)

        assert counts[0] == 0  # 0b00000000
        assert counts[1] == 1  # 0b00000001
        assert counts[2] == 8  # 0b11111111
        assert counts[3] == 4  # 0b10101010

    def test_pack_ternary_batch(self):
        """Test batch ternary packing."""
        from trix.nn.xor_routing import pack_ternary_batch

        batch, dim = 32, 128
        ternary = torch.sign(torch.randn(batch, dim))

        packed = pack_ternary_batch(ternary)

        assert packed.shape == (batch, (dim + 3) // 4)
        assert packed.dtype == torch.uint8

    def test_hamming_distance_batch(self):
        """Test batch Hamming distance computation."""
        from trix.nn.xor_routing import pack_ternary_batch, hamming_distance_batch

        batch = 32
        num_sigs = 16
        dim = 128

        queries = torch.sign(torch.randn(batch, dim))
        sigs = torch.sign(torch.randn(num_sigs, dim))

        packed_queries = pack_ternary_batch(queries)
        packed_sigs = pack_ternary_batch(sigs)

        distances = hamming_distance_batch(packed_queries, packed_sigs)

        assert distances.shape == (batch, num_sigs)
        assert (distances >= 0).all()


class TestCompressionBenchmark:
    """Parametrized compression ratio tests."""

    @pytest.mark.parametrize("num_tiles,d_model,similarity", [
        (16, 256, 0.99),
        (64, 512, 0.99),
        (256, 1024, 0.99),
        (16, 256, 0.95),
        (64, 512, 0.95),
    ])
    def test_compression_ratio(self, num_tiles, d_model, similarity):
        """Test compression ratios for various configurations."""
        from trix.nn.xor_superposition import CompressedSignatures

        # Create base
        base = torch.sign(torch.randn(d_model))

        # Create variations
        flip_prob = 1 - similarity
        signatures = []
        for i in range(num_tiles):
            sig = base.clone()
            flip_mask = torch.rand(d_model) < flip_prob
            sig[flip_mask] *= -1
            signatures.append(sig)
        signatures = torch.stack(signatures)

        compressed = CompressedSignatures().compress(signatures)
        stats = compressed.get_compression_stats()

        # Verify roundtrip
        decompressed = compressed.decompress_all()
        assert torch.equal(signatures, decompressed)

        # Check compression ratio scales with similarity
        if similarity == 0.99:
            assert stats.compression_ratio > 8.0
        elif similarity == 0.95:
            assert stats.compression_ratio > 2.0


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_compression_and_routing(self):
        """Test full compression + routing pipeline."""
        from trix.nn.xor_superposition import XORSuperpositionFFN

        d_model = 128
        num_tiles = 32
        batch = 64

        # Create FFN
        ffn = XORSuperpositionFFN(d_model=d_model, num_tiles=num_tiles)
        x = torch.randn(batch, d_model)

        # Training mode forward
        ffn.train()
        out_train, info_train = ffn(x, return_routing_info=True)

        # Compress for inference
        ffn.compress()
        ffn.eval()
        out_eval, info_eval = ffn(x, return_routing_info=True)

        # Shapes should match
        assert out_train.shape == out_eval.shape

        # Get stats
        stats = ffn.get_compression_stats()
        assert stats is not None
        print(f"\nCompression: {stats.compression_ratio:.1f}x")
        print(f"Mean delta sparsity: {stats.mean_delta_sparsity:.2%}")

    def test_gradient_flow(self):
        """Test gradients flow through FFN."""
        from trix.nn.xor_superposition import XORSuperpositionFFN

        ffn = XORSuperpositionFFN(d_model=64, num_tiles=8)
        x = torch.randn(16, 64, requires_grad=True)

        out, _ = ffn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestEdgeCases:
    """Edge case tests."""

    def test_single_signature(self):
        """Test with single signature."""
        from trix.nn.xor_superposition import CompressedSignatures

        signatures = torch.sign(torch.randn(1, 64))
        compressed = CompressedSignatures().compress(signatures)
        decompressed = compressed.decompress_all()

        assert torch.equal(signatures, decompressed)

    def test_non_divisible_dim(self):
        """Test with dimension not divisible by 4."""
        from trix.nn.xor_superposition import pack_ternary_to_uint8, unpack_uint8_to_ternary

        for dim in [63, 65, 127, 513]:
            ternary = torch.sign(torch.randn(1, dim))
            packed = pack_ternary_to_uint8(ternary)
            unpacked = unpack_uint8_to_ternary(packed, dim)
            assert torch.allclose(ternary, unpacked), f"Failed for dim={dim}"

    def test_all_zeros(self):
        """Test with all-zero signatures."""
        from trix.nn.xor_superposition import CompressedSignatures

        signatures = torch.zeros(8, 64)
        compressed = CompressedSignatures().compress(signatures)
        decompressed = compressed.decompress_all()

        assert torch.equal(signatures, decompressed)

    def test_identical_signatures(self):
        """Test with all identical signatures."""
        from trix.nn.xor_superposition import CompressedSignatures

        base = torch.sign(torch.randn(64))
        signatures = base.unsqueeze(0).expand(16, -1).clone()

        compressed = CompressedSignatures().compress(signatures)
        stats = compressed.get_compression_stats()

        # Should achieve maximum compression
        assert stats.mean_delta_sparsity == 0.0
        assert stats.compression_ratio > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
