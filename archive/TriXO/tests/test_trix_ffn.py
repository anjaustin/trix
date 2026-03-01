"""
Tests for TriX Native Layers (Emergent Routing)

Tests TriXFFN, TriXBlock, and TriXStack with signature-based routing.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn

from trix import TriXFFN, TriXBlock, TriXStack, TriXLinear


class TestTriXLinearSignatures:
    """Tests for signature functionality in TriXLinear."""
    
    def test_get_signature_shape(self):
        """Signature should have shape [in_features]."""
        layer = TriXLinear(64, 256, num_tiles=4)
        sig = layer.get_signature()
        assert sig.shape == (64,)
    
    def test_signature_is_ternary(self):
        """Signature values should be in {-1, 0, +1}."""
        layer = TriXLinear(64, 256, num_tiles=4)
        sig = layer.get_signature()
        unique = sig.unique()
        for val in unique:
            assert val.item() in [-1.0, 0.0, 1.0]
    
    def test_get_tile_signatures_shape(self):
        """Tile signatures should have shape [num_tiles, in_features]."""
        layer = TriXLinear(64, 256, num_tiles=4)
        sigs = layer.get_tile_signatures()
        assert sigs.shape == (4, 64)
    
    def test_tile_signatures_are_ternary(self):
        """All tile signature values should be in {-1, 0, +1}."""
        layer = TriXLinear(64, 256, num_tiles=4)
        sigs = layer.get_tile_signatures()
        unique = sigs.unique()
        for val in unique:
            assert val.item() in [-1.0, 0.0, 1.0]
    
    def test_signatures_cached_after_pack(self):
        """After pack(), signatures should be cached."""
        layer = TriXLinear(64, 256, num_tiles=4)
        layer.pack()
        assert layer.cached_signatures is not None
        assert layer.cached_signatures.shape == (4, 64)
    
    def test_signatures_cleared_after_unpack(self):
        """After unpack(), signature cache should be cleared."""
        layer = TriXLinear(64, 256, num_tiles=4)
        layer.pack()
        layer.unpack()
        assert layer.cached_signatures is None


class TestTriXFFN:
    """Tests for TriXFFN with emergent routing."""
    
    def test_forward_2d(self):
        """Forward should work with 2D input [batch, d_model]."""
        ffn = TriXFFN(64, num_tiles=4)
        x = torch.randn(8, 64)
        out, gate = ffn(x)
        
        assert out.shape == (8, 64)
        assert gate.shape == (8, 4)
    
    def test_forward_3d(self):
        """Forward should work with 3D input [batch, seq, d_model]."""
        ffn = TriXFFN(64, num_tiles=4)
        x = torch.randn(4, 16, 64)
        out, gate = ffn(x)
        
        assert out.shape == (4, 16, 64)
        assert gate.shape == (4, 16, 4)
    
    def test_routing_is_one_hot(self):
        """Routing should be one-hot (winner takes all)."""
        ffn = TriXFFN(64, num_tiles=4)
        x = torch.randn(8, 64)
        _, gate = ffn(x)
        
        # Each row should sum to 1
        assert torch.allclose(gate.sum(dim=-1), torch.ones(8))
    
    def test_routing_is_deterministic(self):
        """Same input should always route to same tile."""
        ffn = TriXFFN(64, num_tiles=4)
        ffn.eval()
        
        x = torch.randn(8, 64)
        _, gate1 = ffn(x)
        _, gate2 = ffn(x)
        
        assert torch.equal(gate1, gate2)
    
    def test_routing_consistency(self):
        """Similar inputs should usually route to same tile."""
        torch.manual_seed(42)
        ffn = TriXFFN(64, num_tiles=4)
        
        base = torch.randn(10, 64)
        perturbed = base + torch.randn_like(base) * 0.01
        
        gate_base = ffn.compute_routing(base)
        gate_pert = ffn.compute_routing(perturbed)
        
        # Most should match
        match_rate = (gate_base.argmax(-1) == gate_pert.argmax(-1)).float().mean()
        assert match_rate > 0.9
    
    def test_routing_discrimination(self):
        """Very different inputs can route to different tiles."""
        torch.manual_seed(42)
        ffn = TriXFFN(64, num_tiles=4)
        
        # Opposite inputs
        x_pos = torch.ones(10, 64)
        x_neg = -torch.ones(10, 64)
        
        gate_pos = ffn.compute_routing(x_pos)
        gate_neg = ffn.compute_routing(x_neg)
        
        # Should have different dominant tiles
        dominant_pos = gate_pos.sum(0).argmax()
        dominant_neg = gate_neg.sum(0).argmax()
        assert dominant_pos != dominant_neg
    
    def test_gradient_flow(self):
        """Gradients should flow through the FFN."""
        ffn = TriXFFN(64, num_tiles=4)
        x = torch.randn(8, 64, requires_grad=True)
        
        out, _ = ffn(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert ffn.up_proj.weight.grad is not None or ffn.up_proj.scales.grad is not None
    
    def test_get_tile_signatures(self):
        """get_tile_signatures should return correct shape."""
        ffn = TriXFFN(64, num_tiles=4)
        sigs = ffn.get_tile_signatures()
        assert sigs.shape == (4, 64)
    
    def test_get_routing_stats(self):
        """get_routing_stats should return expected keys."""
        ffn = TriXFFN(64, num_tiles=4)
        x = torch.randn(100, 64)
        
        stats = ffn.get_routing_stats(x)
        
        assert 'tile_usage' in stats
        assert 'balance' in stats
        assert 'entropy' in stats
        assert 'dominant_tile' in stats
        assert len(stats['tile_usage']) == 4
    
    def test_get_signature_diversity(self):
        """get_signature_diversity should return value in [0, 1]."""
        ffn = TriXFFN(64, num_tiles=4)
        diversity = ffn.get_signature_diversity()
        
        assert 0.0 <= diversity <= 1.0
    
    def test_pack_unpack(self):
        """pack() and unpack() should work correctly."""
        ffn = TriXFFN(64, num_tiles=4)
        x = torch.randn(8, 64)
        
        # Pack for inference
        ffn.eval()
        ffn.pack()
        
        # Should work without error
        out_packed, gate = ffn(x)
        assert out_packed.shape == (8, 64)
        
        # Signatures should be cached
        assert ffn.up_proj.cached_signatures is not None
        
        # Unpack should clear caches
        ffn.unpack()
        assert ffn.up_proj.cached_signatures is None


class TestTriXBlock:
    """Tests for TriXBlock transformer block."""
    
    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        block = TriXBlock(64, n_heads=4, num_tiles=4)
        x = torch.randn(4, 16, 64)
        
        out, gate = block(x)
        
        assert out.shape == (4, 16, 64)
        assert gate.shape == (4, 16, 4)
    
    def test_causal_masking(self):
        """Causal masking should work without errors."""
        block = TriXBlock(64, n_heads=4, num_tiles=4)
        x = torch.randn(4, 16, 64)
        
        out, _ = block(x, is_causal=True)
        assert out.shape == (4, 16, 64)
    
    def test_residual_connection(self):
        """Output should have correlation with input (residual)."""
        block = TriXBlock(64, n_heads=4, num_tiles=4)
        x = torch.randn(4, 16, 64)
        
        out, _ = block(x)
        
        # Flatten and check correlation
        corr = torch.corrcoef(torch.stack([x.flatten(), out.flatten()]))[0, 1]
        assert corr > 0.0
    
    def test_gradient_flow(self):
        """Gradients should flow through the block."""
        block = TriXBlock(64, n_heads=4, num_tiles=4)
        x = torch.randn(4, 16, 64, requires_grad=True)
        
        out, _ = block(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert block.attention.in_proj_weight.grad is not None
    
    def test_get_routing_stats(self):
        """get_routing_stats should work."""
        block = TriXBlock(64, n_heads=4, num_tiles=4)
        x = torch.randn(4, 16, 64)
        
        stats = block.get_routing_stats(x)
        assert 'tile_usage' in stats


class TestTriXStack:
    """Tests for TriXStack (multiple blocks)."""
    
    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        stack = TriXStack(n_layers=3, d_model=64, n_heads=4, num_tiles=4)
        x = torch.randn(4, 16, 64)
        
        out, all_routing = stack(x)
        
        assert out.shape == (4, 16, 64)
        assert len(all_routing) == 3
        for routing in all_routing:
            assert routing.shape == (4, 16, 4)
    
    def test_causal_masking(self):
        """Causal masking should work."""
        stack = TriXStack(n_layers=3, d_model=64, n_heads=4, num_tiles=4)
        x = torch.randn(4, 16, 64)
        
        out, _ = stack(x, is_causal=True)
        assert out.shape == (4, 16, 64)
    
    def test_get_all_routing_stats(self):
        """get_all_routing_stats should return stats for each block."""
        stack = TriXStack(n_layers=3, d_model=64, n_heads=4, num_tiles=4)
        x = torch.randn(4, 16, 64)
        
        all_stats = stack.get_all_routing_stats(x)
        
        assert len(all_stats) == 3
        for i, stats in enumerate(all_stats):
            assert stats['block'] == i
            assert 'tile_usage' in stats
    
    def test_pack_unpack(self):
        """pack() and unpack() should work for all blocks."""
        stack = TriXStack(n_layers=3, d_model=64, n_heads=4, num_tiles=4)
        
        stack.pack()
        for block in stack.blocks:
            assert block.ffn.up_proj._packed
        
        stack.unpack()
        for block in stack.blocks:
            assert not block.ffn.up_proj._packed


class TestRoutingStability:
    """Tests for routing stability during training."""
    
    def test_routing_stabilizes(self):
        """Routing should stabilize as training progresses."""
        torch.manual_seed(42)
        
        ffn = TriXFFN(64, num_tiles=4, dropout=0.0)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        x = torch.randn(50, 64)
        target = torch.randn(50, 64)
        
        # Track routing changes
        prev_routing = None
        changes = []
        
        for step in range(50):
            optimizer.zero_grad()
            out, _ = ffn(x)
            loss = nn.functional.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                gate = ffn.compute_routing(x)
                if prev_routing is not None:
                    change = (gate.argmax(-1) != prev_routing.argmax(-1)).float().mean()
                    changes.append(change.item())
                prev_routing = gate.clone()
        
        # Later changes should be smaller than early changes
        early_changes = sum(changes[:10]) / 10
        late_changes = sum(changes[-10:]) / 10
        assert late_changes <= early_changes


class TestIdealInputRouting:
    """Test that ideal inputs route to their corresponding tiles."""
    
    def test_signature_routes_to_self(self):
        """Each tile's signature should route to that tile."""
        ffn = TriXFFN(64, num_tiles=4)
        sigs = ffn.get_tile_signatures()
        
        for t in range(4):
            ideal_input = sigs[t].unsqueeze(0)  # [1, d_model]
            gate = ffn.compute_routing(ideal_input)
            winner = gate.argmax().item()
            assert winner == t, f"Tile {t}'s ideal input routed to tile {winner}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
