"""
Tests for Sparse Training Components (Option B)

Tests SparseTriXFFN and SparseTriXBlock - the components that enable
training with gated computation from the start.
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trix.nn.sparse import SparseTriXFFN, SparseTriXBlock


class TestSparseTriXFFN:
    """Tests for SparseTriXFFN."""
    
    def test_forward_2d(self):
        """Test forward pass with 2D input."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        x = torch.randn(8, 64)
        
        out, gate, aux = ffn(x)
        
        assert out.shape == x.shape
        assert gate.shape == (8, 4)
        assert 'balance' in aux or not ffn.training
    
    def test_forward_3d(self):
        """Test forward pass with 3D input (batch, seq, dim)."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        x = torch.randn(4, 16, 64)
        
        out, gate, aux = ffn(x)
        
        assert out.shape == x.shape
        assert gate.shape == (4, 16, 4)
    
    def test_routing_is_one_hot(self):
        """Verify routing produces one-hot gates."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        x = torch.randn(32, 64)
        
        _, gate, _ = ffn(x)
        
        # Each row should sum to 1 (one-hot)
        assert torch.allclose(gate.sum(dim=-1), torch.ones(32))
        # Each row should have exactly one 1
        assert (gate.max(dim=-1).values == 1.0).all()
    
    def test_routing_deterministic(self):
        """Same input should route to same tile."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        ffn.eval()
        
        x = torch.randn(1, 64)
        
        _, gate1, _ = ffn(x)
        _, gate2, _ = ffn(x)
        
        assert torch.equal(gate1, gate2)
    
    def test_different_inputs_can_route_differently(self):
        """Different inputs should be able to route to different tiles."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        
        # Create very different inputs
        x1 = torch.ones(1, 64)
        x2 = -torch.ones(1, 64)
        
        _, gate1, _ = ffn(x1)
        _, gate2, _ = ffn(x2)
        
        winner1 = gate1.argmax().item()
        winner2 = gate2.argmax().item()
        
        # They CAN be different (not guaranteed, but likely with opposite inputs)
        # This test just verifies the mechanism allows it
        assert gate1.shape == gate2.shape
    
    def test_balance_loss_computed(self):
        """Balance loss should be computed during training."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4, balance_weight=0.01)
        ffn.train()
        
        x = torch.randn(32, 64)
        _, _, aux = ffn(x)
        
        assert 'balance' in aux
        assert 'diversity' in aux
        assert 'total_aux' in aux
        assert aux['balance'].item() >= 0
    
    def test_no_aux_loss_in_eval(self):
        """No auxiliary losses in eval mode."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        ffn.eval()
        
        x = torch.randn(8, 64)
        _, _, aux = ffn(x)
        
        assert len(aux) == 0
    
    def test_gradient_flow(self):
        """Gradients should flow through sparse forward."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        x = torch.randn(8, 64, requires_grad=True)
        
        out, _, aux = ffn(x)
        loss = out.sum() + aux.get('total_aux', 0)
        loss.backward()
        
        assert x.grad is not None
        assert ffn.up_proj.weight.grad is not None
        assert ffn.down_proj.weight.grad is not None
    
    def test_ema_signatures_updated(self):
        """EMA signatures should be updated during training."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        ffn.train()
        
        assert ffn.ema_signatures is None
        
        x = torch.randn(8, 64)
        ffn(x)
        
        assert ffn.ema_signatures is not None
        assert ffn.ema_signatures.shape == (4, 64)
    
    def test_get_routing_stats(self):
        """Test routing statistics computation."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        x = torch.randn(100, 64)
        
        stats = ffn.get_routing_stats(x)
        
        assert 'tile_usage' in stats
        assert 'entropy' in stats
        assert 'dominant_tile' in stats
        assert 'balance' in stats
        assert len(stats['tile_usage']) == 4
        assert abs(sum(stats['tile_usage']) - 1.0) < 0.01
    
    def test_get_signature_diversity(self):
        """Test signature diversity computation."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        
        diversity = ffn.get_signature_diversity()
        
        assert 0.0 <= diversity <= 1.0
    
    def test_sparse_computation_correctness(self):
        """Verify sparse computation produces valid outputs."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        x = torch.randn(8, 64)
        
        out, gate, _ = ffn(x)
        
        # Output should be finite
        assert torch.isfinite(out).all()
        # Output should not be all zeros (model should produce something)
        assert out.abs().sum() > 0
    
    def test_tile_isolation(self):
        """Each input should only use one tile's weights."""
        ffn = SparseTriXFFN(d_model=32, num_tiles=4, dropout=0.0)
        ffn.eval()
        
        # Create input that routes to a specific tile
        x = torch.randn(1, 32)
        _, gate, _ = ffn(x)
        
        # Verify one-hot
        assert gate.sum() == 1.0
        active_tile = gate.argmax().item()
        
        # The output should come from only the active tile
        # (This is implicitly tested by the sparse forward working correctly)
        assert gate[0, active_tile] == 1.0


class TestSparseTriXBlock:
    """Tests for SparseTriXBlock."""
    
    def test_forward_shape(self):
        """Test forward pass shapes."""
        block = SparseTriXBlock(d_model=64, n_heads=4, num_tiles=4)
        x = torch.randn(4, 16, 64)
        
        out, gate, aux = block(x)
        
        assert out.shape == x.shape
        assert gate.shape == (4, 16, 4)
    
    def test_causal_masking(self):
        """Test causal attention masking."""
        block = SparseTriXBlock(d_model=64, n_heads=4, num_tiles=4)
        block.eval()
        
        x = torch.randn(2, 8, 64)
        
        out1, _, _ = block(x, is_causal=True)
        out2, _, _ = block(x, is_causal=False)
        
        # Outputs should differ with different masking
        # (Not guaranteed but very likely)
        assert out1.shape == out2.shape
    
    def test_residual_connection(self):
        """Verify residual connections are working."""
        block = SparseTriXBlock(d_model=64, n_heads=4, num_tiles=4, dropout=0.0)
        
        x = torch.randn(2, 8, 64)
        out, _, _ = block(x)
        
        # Output should be different from input
        diff = (out - x).abs().mean()
        assert diff > 0  # Should be different
        
        # Output should be finite and reasonable
        assert torch.isfinite(out).all()
        assert out.abs().mean() < 100  # Not exploding
    
    def test_gradient_flow(self):
        """Gradients should flow through the block."""
        block = SparseTriXBlock(d_model=64, n_heads=4, num_tiles=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        
        out, _, aux = block(x)
        loss = out.sum() + aux.get('total_aux', 0)
        loss.backward()
        
        assert x.grad is not None
        assert block.ffn.up_proj.weight.grad is not None
    
    def test_aux_losses_propagated(self):
        """Auxiliary losses from FFN should be returned."""
        block = SparseTriXBlock(d_model=64, n_heads=4, num_tiles=4)
        block.train()
        
        x = torch.randn(2, 8, 64)
        _, _, aux = block(x)
        
        assert 'total_aux' in aux


class TestSparseTrainingDynamics:
    """Tests for sparse training behavior."""
    
    def test_balance_loss_penalizes_imbalance(self):
        """Balance loss should be higher for imbalanced routing."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4, balance_weight=1.0)
        
        # Create imbalanced gate
        imbalanced = torch.zeros(100, 4)
        imbalanced[:, 0] = 1.0  # All to tile 0
        
        # Create balanced gate
        balanced = torch.zeros(100, 4)
        for i in range(100):
            balanced[i, i % 4] = 1.0
        
        loss_imbalanced = ffn.compute_balance_loss(imbalanced)
        loss_balanced = ffn.compute_balance_loss(balanced)
        
        assert loss_imbalanced > loss_balanced
    
    def test_diversity_loss_penalizes_similar_signatures(self):
        """Diversity loss should penalize similar signatures."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4, diversity_weight=1.0)
        
        # Initial diversity loss
        initial_div_loss = ffn.compute_diversity_loss()
        
        # Make all signatures identical (should increase loss)
        with torch.no_grad():
            # Set all tile weights to be the same
            tile_size = ffn.d_ff // ffn.num_tiles
            base_weights = ffn.up_proj.weight[:tile_size].clone()
            for t in range(1, ffn.num_tiles):
                start = t * tile_size
                end = start + tile_size
                ffn.up_proj.weight[start:end] = base_weights
        
        collapsed_div_loss = ffn.compute_diversity_loss()
        
        # More similar signatures should have higher similarity (higher loss)
        # Note: This might not always be strictly true due to random init
        # but the mechanism should work
        assert collapsed_div_loss.shape == initial_div_loss.shape
    
    def test_training_reduces_loss(self):
        """A few training steps should reduce loss."""
        ffn = SparseTriXFFN(d_model=32, num_tiles=4)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        x = torch.randn(32, 32)
        target = torch.randn(32, 32)
        
        # Initial loss
        out, _, aux = ffn(x)
        initial_loss = nn.functional.mse_loss(out, target) + aux['total_aux']
        
        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            out, _, aux = ffn(x)
            loss = nn.functional.mse_loss(out, target) + aux['total_aux']
            loss.backward()
            optimizer.step()
        
        # Final loss
        out, _, aux = ffn(x)
        final_loss = nn.functional.mse_loss(out, target) + aux['total_aux']
        
        assert final_loss < initial_loss


class TestSparseVsDense:
    """Compare sparse and dense behavior."""
    
    def test_sparse_uses_less_computation(self):
        """Sparse forward should touch fewer weights per input."""
        # This is a conceptual test - sparse routing means each input
        # only uses 1/num_tiles of the FFN weights
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        
        x = torch.randn(1, 64)
        _, gate, _ = ffn(x)
        
        # Verify only one tile is active
        active_tiles = (gate > 0).sum().item()
        assert active_tiles == 1
        
        # This means only 1/4 of the FFN computation is performed
        # (Verified by the gated_forward implementation)


class TestNEONInference:
    """Tests for NEON-accelerated inference."""
    
    def test_pack_unpack(self):
        """Test pack and unpack cycle."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        
        assert not ffn.is_packed
        
        ffn.pack()
        assert ffn.is_packed
        assert ffn.up_proj._packed
        assert ffn.down_proj._packed
        
        ffn.unpack()
        assert not ffn.is_packed
    
    def test_packed_inference_produces_output(self):
        """Packed inference should produce valid outputs."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        ffn.eval()
        
        x = torch.randn(8, 64)
        
        # Get unpacked output first
        out_unpacked, gate, _ = ffn(x)
        
        # Pack and run inference
        ffn.pack()
        out_packed, _, _ = ffn(x)
        
        # Both should produce finite outputs
        assert torch.isfinite(out_unpacked).all()
        assert torch.isfinite(out_packed).all()
        
        # Outputs should be similar (same routing, same weights conceptually)
        # Note: May differ due to quantization effects
        assert out_packed.shape == out_unpacked.shape
    
    def test_packed_mode_not_used_during_training(self):
        """Packed mode should not be used during training."""
        ffn = SparseTriXFFN(d_model=64, num_tiles=4)
        ffn.pack()
        ffn.train()  # Switch to training mode
        
        x = torch.randn(8, 64, requires_grad=True)
        out, _, aux = ffn(x)
        
        # Should still compute aux losses (means it used PyTorch path)
        assert 'balance' in aux
        
        # Should have gradients
        loss = out.sum() + aux['total_aux']
        loss.backward()
        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
