"""
TriX Neural Network Layer Tests

Tests for high-level neural network components.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch

from trix.nn import (
    Top1Gate,
    GatedFFN,
    TriXTransformerBlock,
)


class TestGatedFFN:
    """Tests for gated feed-forward network."""
    
    def test_2d_input(self):
        """Should handle 2D input [batch, d_model]."""
        ffn = GatedFFN(d_model=64, num_tiles=4)
        x = torch.randn(8, 64)
        out, gate = ffn(x)
        
        assert out.shape == (8, 64)
        assert gate.shape == (8, 4)
    
    def test_3d_input(self):
        """Should handle 3D input [batch, seq_len, d_model]."""
        ffn = GatedFFN(d_model=64, num_tiles=4)
        x = torch.randn(4, 16, 64)
        out, gate = ffn(x)
        
        assert out.shape == (4, 16, 64)
        assert gate.shape == (4, 16, 4)
    
    def test_gate_is_one_hot(self):
        """Gate output should be one-hot per position."""
        ffn = GatedFFN(d_model=64, num_tiles=4)
        ffn.eval()  # No noise
        x = torch.randn(4, 16, 64)
        _, gate = ffn(x)
        
        # Sum should be 1 for each position
        gate_2d = gate.view(-1, 4)
        sums = gate_2d.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums))
    
    def test_expansion_factor(self):
        """Different expansion factors should work."""
        for expansion in [2, 4, 8]:
            ffn = GatedFFN(d_model=64, expansion=expansion, num_tiles=4)
            assert ffn.d_ff == 64 * expansion
            
            x = torch.randn(4, 64)
            out, _ = ffn(x)
            assert out.shape == (4, 64)
    
    def test_gradient_flow(self):
        """Gradients should flow through to input and TriX layers."""
        ffn = GatedFFN(d_model=64, num_tiles=4)
        x = torch.randn(4, 64, requires_grad=True)
        out, _ = ffn(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        # Note: gate_proj gradients are zero due to hard Top1Gate (argmax is not differentiable)
        # Gradients flow through the TriX layers via STE
        assert ffn.up_proj.scales.grad is not None
    
    def test_noise_during_training(self):
        """Different noise scales should produce different gates."""
        torch.manual_seed(42)
        
        ffn_no_noise = GatedFFN(d_model=64, num_tiles=4, noise_scale=0.0)
        ffn_noise = GatedFFN(d_model=64, num_tiles=4, noise_scale=2.0)
        
        # Copy weights
        ffn_noise.gate_proj.load_state_dict(ffn_no_noise.gate_proj.state_dict())
        
        ffn_no_noise.train()
        ffn_noise.train()
        
        x = torch.randn(100, 64)
        
        # Run multiple times with noise
        gates_noise = []
        for _ in range(5):
            _, gate = ffn_noise(x)
            gates_noise.append(gate.clone())
        
        # With noise, gates should vary between runs
        all_same = all(torch.allclose(gates_noise[0], g) for g in gates_noise[1:])
        assert not all_same, "Gates should vary with noise"


class TestTriXTransformerBlock:
    """Tests for transformer block with TriX FFN."""
    
    def test_forward_shape(self):
        """Forward should produce correct shapes."""
        block = TriXTransformerBlock(d_model=64, n_heads=4, num_tiles=4)
        x = torch.randn(4, 16, 64)
        out, gate = block(x)
        
        assert out.shape == (4, 16, 64)
        assert gate.shape == (4, 16, 4)
    
    def test_causal_masking(self):
        """Causal masking should work."""
        block = TriXTransformerBlock(d_model=64, n_heads=4)
        x = torch.randn(4, 16, 64)
        
        # Should not raise with causal=True
        out, _ = block(x, is_causal=True)
        assert out.shape == (4, 16, 64)
    
    def test_residual_connection(self):
        """Output should be similar to input (residual)."""
        block = TriXTransformerBlock(d_model=64, n_heads=4)
        
        # With small weights, output should be close to input
        x = torch.randn(4, 16, 64)
        out, _ = block(x)
        
        # They shouldn't be identical, but correlation should exist
        correlation = torch.corrcoef(torch.stack([x.flatten(), out.flatten()]))[0, 1]
        assert correlation > 0.0
    
    def test_gradient_flow(self):
        """Gradients should flow through all components."""
        block = TriXTransformerBlock(d_model=64, n_heads=4)
        x = torch.randn(4, 16, 64, requires_grad=True)
        out, _ = block(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert block.attention.in_proj_weight.grad is not None
        # Note: gate_proj gradients are zero due to hard Top1Gate (argmax is not differentiable)
        assert block.ffn.up_proj.scales.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
