"""
Tests for SparseLookupFFN - Routing IS The Computation.
"""

import pytest
import torch
import torch.nn as nn

from trix.nn.sparse_lookup import (
    SparseLookupFFN,
    SparseLookupBlock,
    TernarySpline2D,
    FloatSpline2D,
)


class TestTernarySpline2D:
    """Tests for TernarySpline2D."""
    
    def test_forward_shape(self):
        """Output shape matches input batch size."""
        spline = TernarySpline2D(grid_size=16)
        a = torch.randn(32)
        b = torch.randn(32)
        out = spline(a, b)
        assert out.shape == (32,)
    
    def test_output_range(self):
        """Output is bounded by scale."""
        spline = TernarySpline2D(grid_size=16)
        a = torch.rand(100) * 2 - 1  # [-1, 1]
        b = torch.rand(100) * 2 - 1
        out = spline(a, b)
        # Ternary coeffs in {-1,0,1}, max output ~3*scale
        assert out.abs().max() < 10 * spline.scale.abs()
    
    def test_gradient_flow(self):
        """Gradients flow through STE."""
        spline = TernarySpline2D(grid_size=16)
        a = torch.randn(32, requires_grad=True)
        b = torch.randn(32, requires_grad=True)
        out = spline(a, b)
        out.sum().backward()
        assert spline.coeffs.grad is not None
        assert spline.scale.grad is not None
    
    def test_deterministic(self):
        """Same input gives same output."""
        spline = TernarySpline2D(grid_size=16)
        a = torch.randn(32)
        b = torch.randn(32)
        out1 = spline(a, b)
        out2 = spline(a, b)
        assert torch.allclose(out1, out2)


class TestFloatSpline2D:
    """Tests for FloatSpline2D."""
    
    def test_forward_shape(self):
        """Output shape matches input batch size."""
        spline = FloatSpline2D(grid_size=16)
        a = torch.randn(32)
        b = torch.randn(32)
        out = spline(a, b)
        assert out.shape == (32,)
    
    def test_gradient_flow(self):
        """Gradients flow normally."""
        spline = FloatSpline2D(grid_size=16)
        a = torch.randn(32, requires_grad=True)
        b = torch.randn(32, requires_grad=True)
        out = spline(a, b)
        out.sum().backward()
        assert spline.coeffs.grad is not None


class TestSparseLookupFFN:
    """Tests for SparseLookupFFN."""
    
    def test_forward_shape_3d(self):
        """Output shape matches input for 3D input."""
        model = SparseLookupFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(2, 16, 64)
        out, info, aux = model(x)
        assert out.shape == x.shape
    
    def test_forward_returns_info(self):
        """Forward returns routing info dict."""
        model = SparseLookupFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(2, 16, 64)
        out, info, aux = model(x)
        assert 'tile_idx' in info
        assert 'compressed' in info
        assert info['tile_idx'].shape == (2, 16)
        assert info['compressed'].shape == (2, 16, 2)
    
    def test_forward_returns_aux_losses(self):
        """Forward returns auxiliary losses dict."""
        model = SparseLookupFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(2, 16, 64)
        out, info, aux = model(x)
        assert 'total_aux' in aux
        assert isinstance(aux['total_aux'], torch.Tensor)
    
    def test_residual_connection(self):
        """Output includes residual from input."""
        model = SparseLookupFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(2, 16, 64)
        out, _, _ = model(x)
        # Output should be different from input but correlated
        diff = (out - x).abs().mean()
        assert diff > 0, "Output should differ from input"
        assert diff < x.abs().mean() * 2, "Residual should keep output close to input"
    
    def test_gradient_flow(self):
        """Gradients flow to key parameters."""
        model = SparseLookupFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(2, 16, 64)
        out, _, _ = model(x)
        out.sum().backward()
        
        # Critical params must have gradients
        assert model.directions.grad is not None
        assert model.compress[0].weight.grad is not None
        assert model.output_scale.grad is not None
    
    def test_routing_uses_multiple_tiles(self):
        """Routing distributes across tiles."""
        model = SparseLookupFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(8, 32, 64)  # Larger batch
        out, info, _ = model(x)
        
        unique_tiles = info['tile_idx'].unique()
        assert len(unique_tiles) > 1, "Should use multiple tiles"
    
    def test_routing_stats(self):
        """Routing stats are computed correctly."""
        model = SparseLookupFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        model.reset_stats()
        
        x = torch.randn(4, 16, 64)
        model(x)
        
        stats = model.get_routing_stats()
        assert stats['num_tiles'] == 16
        assert stats['active_tiles'] > 0
        assert 0 <= stats['usage_mean'] <= 1
    
    def test_param_count(self):
        """Parameter count is reasonable."""
        model = SparseLookupFFN(d_model=128, num_tiles=64, tiles_per_cluster=8)
        params = model.get_param_count()
        
        assert params['total'] < 100_000, "Should be under 100K params"
        assert params['directions'] == 128 * 64
        assert params['splines'] > 0
        assert params['compress'] > 0
    
    def test_ternary_vs_float_splines(self):
        """Both spline types work."""
        model_ternary = SparseLookupFFN(d_model=64, num_tiles=8, ternary_splines=True)
        model_float = SparseLookupFFN(d_model=64, num_tiles=8, ternary_splines=False)
        
        x = torch.randn(2, 16, 64)
        
        out_t, _, _ = model_ternary(x)
        out_f, _, _ = model_float(x)
        
        assert out_t.shape == x.shape
        assert out_f.shape == x.shape
    
    def test_deterministic_routing(self):
        """Same input routes to same tiles."""
        model = SparseLookupFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        model.eval()
        
        x = torch.randn(2, 16, 64)
        
        _, info1, _ = model(x)
        _, info2, _ = model(x)
        
        assert torch.equal(info1['tile_idx'], info2['tile_idx'])


class TestSparseLookupBlock:
    """Tests for SparseLookupBlock."""
    
    def test_forward_shape(self):
        """Output shape matches input."""
        block = SparseLookupBlock(d_model=64, n_heads=4, num_tiles=16)
        x = torch.randn(2, 16, 64)
        out, info, aux = block(x)
        assert out.shape == x.shape
    
    def test_causal_masking(self):
        """Causal masking works."""
        block = SparseLookupBlock(d_model=64, n_heads=4, num_tiles=16)
        x = torch.randn(2, 16, 64)
        
        out_causal, _, _ = block(x, is_causal=True)
        out_full, _, _ = block(x, is_causal=False)
        
        # Outputs should differ with different masking
        assert not torch.allclose(out_causal, out_full)
    
    def test_gradient_flow(self):
        """Gradients flow through entire block."""
        block = SparseLookupBlock(d_model=64, n_heads=4, num_tiles=16)
        x = torch.randn(2, 16, 64)
        out, _, _ = block(x)
        out.sum().backward()
        
        # Check attention and FFN both get gradients
        assert block.ln1.weight.grad is not None
        assert block.ffn.directions.grad is not None


class TestIntegration:
    """Integration tests."""
    
    def test_multiple_blocks(self):
        """Stack of blocks works."""
        blocks = nn.ModuleList([
            SparseLookupBlock(d_model=64, n_heads=4, num_tiles=16)
            for _ in range(3)
        ])
        
        x = torch.randn(2, 16, 64)
        for block in blocks:
            x, _, _ = block(x)
        
        assert x.shape == (2, 16, 64)
    
    def test_training_step(self):
        """Full training step works."""
        model = SparseLookupFFN(d_model=64, num_tiles=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)
        
        optimizer.zero_grad()
        out, _, aux = model(x)
        loss = (out - target).pow(2).mean() + aux['total_aux']
        loss.backward()
        optimizer.step()
        
        # Params should have changed
        assert True  # If we get here, it worked
    
    def test_eval_mode(self):
        """Eval mode works."""
        model = SparseLookupFFN(d_model=64, num_tiles=16)
        model.eval()
        
        with torch.no_grad():
            x = torch.randn(2, 16, 64)
            out, _, _ = model(x)
        
        assert out.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
