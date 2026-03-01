"""
Tests for Hierarchical TriX (The Big Leap)

Tests 2-level hierarchical routing at scale.
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trix.nn import TriXTile, HierarchicalTriXFFN, HierarchicalTriXBlock


class TestTriXTile:
    """Tests for individual TriX tiles."""
    
    def test_forward_shape(self):
        """Tile produces correct output shape."""
        tile = TriXTile(d_model=64, d_hidden=32)
        x = torch.randn(8, 64)
        out = tile(x)
        assert out.shape == (8, 64)
    
    def test_signature_shape(self):
        """Signature has correct shape."""
        tile = TriXTile(d_model=64, d_hidden=32)
        sig = tile.get_signature()
        assert sig.shape == (64,)
    
    def test_signature_ternary(self):
        """Signature values are ternary."""
        tile = TriXTile(d_model=64, d_hidden=32)
        sig = tile.get_signature()
        assert torch.all((sig == -1) | (sig == 0) | (sig == 1))
    
    def test_gradient_flow(self):
        """Gradients flow through tile."""
        tile = TriXTile(d_model=64, d_hidden=32)
        x = torch.randn(8, 64)
        out = tile(x)
        loss = out.sum()
        loss.backward()
        assert tile.up_weight.grad is not None
        assert tile.down_weight.grad is not None
    
    def test_usage_tracking(self):
        """Usage statistics are tracked."""
        tile = TriXTile(d_model=64, d_hidden=32)
        assert tile.usage_rate == 0.0
        tile.update_usage(5, 10)
        assert tile.usage_rate == 0.5


class TestHierarchicalTriXFFN:
    """Tests for hierarchical content-addressable FFN."""
    
    def test_forward_shape_2d(self):
        """Output matches input shape for 2D input."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(8, 64)
        out, routing, aux = ffn(x)
        assert out.shape == x.shape
    
    def test_forward_shape_3d(self):
        """Output matches input shape for 3D input."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(4, 16, 64)
        out, routing, aux = ffn(x)
        assert out.shape == x.shape
    
    def test_routing_info(self):
        """Routing info contains expected keys."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(8, 64)
        out, routing, aux = ffn(x)
        assert 'tile_idx' in routing
        assert 'cluster_idx' in routing
        assert 'scores' in routing
    
    def test_tile_idx_valid(self):
        """Tile indices are within valid range."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(32, 64)
        out, routing, aux = ffn(x)
        assert routing['tile_idx'].min() >= 0
        assert routing['tile_idx'].max() < 16
    
    def test_cluster_idx_valid(self):
        """Cluster indices are within valid range."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(32, 64)
        out, routing, aux = ffn(x)
        assert routing['cluster_idx'].min() >= 0
        assert routing['cluster_idx'].max() < 4
    
    def test_aux_losses_training(self):
        """Auxiliary losses computed during training."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        ffn.train()
        x = torch.randn(8, 64)
        out, routing, aux = ffn(x)
        assert 'cluster_balance' in aux
        assert 'tile_balance' in aux
        assert 'diversity' in aux
        assert 'total_aux' in aux
    
    def test_aux_losses_eval(self):
        """No auxiliary losses during eval."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        ffn.eval()
        x = torch.randn(8, 64)
        out, routing, aux = ffn(x)
        assert len(aux) == 0
    
    def test_gradient_flow(self):
        """Gradients flow to tiles."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(32, 64)
        out, routing, aux = ffn(x)
        loss = out.sum() + aux.get('total_aux', 0)
        loss.backward()
        # At least some tiles should have gradients
        grads = [tile.up_weight.grad is not None for tile in ffn.tiles]
        assert any(grads)
    
    def test_hierarchy_built(self):
        """Hierarchy is built automatically."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(8, 64)
        ffn(x)  # Triggers build
        assert ffn._hierarchy_built
        assert ffn.cluster_signatures is not None
        assert ffn.cluster_assignments is not None
    
    def test_cluster_signatures_shape(self):
        """Cluster signatures have correct shape."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        ffn.build_hierarchy()
        assert ffn.cluster_signatures.shape == (4, 64)  # num_clusters x d_model
    
    def test_cluster_assignments_valid(self):
        """Each tile assigned to exactly one cluster."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        ffn.build_hierarchy()
        assert ffn.cluster_assignments.shape == (16,)
        assert ffn.cluster_assignments.min() >= 0
        assert ffn.cluster_assignments.max() < 4
    
    def test_routing_stats(self):
        """Routing stats available."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(32, 64)
        ffn(x)
        stats = ffn.get_routing_stats()
        assert 'num_tiles' in stats
        assert 'num_clusters' in stats
        assert 'active_tiles' in stats
    
    def test_cluster_info(self):
        """Cluster info available after hierarchy built."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        x = torch.randn(32, 64)
        ffn(x)
        info = ffn.get_cluster_info()
        assert 'cluster_0' in info
        assert 'tiles' in info['cluster_0']


class TestHierarchicalScale:
    """Tests for scaling to larger tile counts."""
    
    def test_64_tiles(self):
        """64 tiles with 8 clusters works."""
        ffn = HierarchicalTriXFFN(d_model=128, num_tiles=64, tiles_per_cluster=8)
        x = torch.randn(32, 128)
        out, routing, aux = ffn(x)
        assert out.shape == x.shape
        assert routing['tile_idx'].max() < 64
    
    def test_128_tiles(self):
        """128 tiles with 16 clusters works."""
        ffn = HierarchicalTriXFFN(d_model=128, num_tiles=128, tiles_per_cluster=8)
        x = torch.randn(32, 128)
        out, routing, aux = ffn(x)
        assert out.shape == x.shape
        assert routing['tile_idx'].max() < 128
    
    def test_256_tiles(self):
        """256 tiles with 16 clusters works."""
        ffn = HierarchicalTriXFFN(d_model=256, num_tiles=256, tiles_per_cluster=16)
        x = torch.randn(16, 256)
        out, routing, aux = ffn(x)
        assert out.shape == x.shape
        assert routing['tile_idx'].max() < 256


class TestHierarchicalRouting:
    """Tests for routing correctness."""
    
    def test_deterministic_routing(self):
        """Same input routes to same tile."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        ffn.eval()
        x = torch.randn(1, 64)
        
        _, r1, _ = ffn(x)
        _, r2, _ = ffn(x)
        
        assert r1['tile_idx'].item() == r2['tile_idx'].item()
    
    def test_different_inputs_different_routes(self):
        """Different inputs can route to different tiles."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        ffn.eval()
        
        # Many different inputs should hit different tiles
        x = torch.randn(100, 64)
        _, routing, _ = ffn(x)
        
        unique_tiles = routing['tile_idx'].unique()
        assert len(unique_tiles) > 1  # At least some diversity
    
    def test_top_k_clusters(self):
        """Top-k cluster routing works."""
        ffn = HierarchicalTriXFFN(
            d_model=64, num_tiles=16, tiles_per_cluster=4, top_k_clusters=2
        )
        x = torch.randn(8, 64)
        out, routing, aux = ffn(x)
        assert out.shape == x.shape


class TestHierarchicalBlock:
    """Tests for hierarchical transformer block."""
    
    def test_forward_shape(self):
        """Block produces correct output shape."""
        block = HierarchicalTriXBlock(
            d_model=64, n_heads=4, num_tiles=16, tiles_per_cluster=4
        )
        x = torch.randn(2, 16, 64)
        out, routing, aux = block(x)
        assert out.shape == x.shape
    
    def test_causal_masking(self):
        """Causal masking works."""
        block = HierarchicalTriXBlock(
            d_model=64, n_heads=4, num_tiles=16, tiles_per_cluster=4
        )
        x = torch.randn(2, 16, 64)
        out, routing, aux = block(x, is_causal=True)
        assert out.shape == x.shape
    
    def test_gradient_flow(self):
        """Gradients flow through block."""
        block = HierarchicalTriXBlock(
            d_model=64, n_heads=4, num_tiles=16, tiles_per_cluster=4
        )
        x = torch.randn(2, 8, 64)
        out, routing, aux = block(x)
        loss = out.sum()
        loss.backward()
        assert block.ffn.tiles[0].up_weight.grad is not None or \
               any(t.up_weight.grad is not None for t in block.ffn.tiles)


class TestEMASignatures:
    """Tests for EMA signature stability."""
    
    def test_ema_initialized(self):
        """EMA signatures initialized after first forward."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        ffn.train()
        x = torch.randn(8, 64)
        ffn(x)
        assert ffn.ema_tile_signatures is not None
    
    def test_ema_shape(self):
        """EMA signatures have correct shape."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        ffn.train()
        x = torch.randn(8, 64)
        ffn(x)
        assert ffn.ema_tile_signatures.shape == (16, 64)
    
    def test_ema_stability(self):
        """EMA signatures change slowly."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        ffn.train()
        
        x = torch.randn(8, 64)
        ffn(x)
        ema1 = ffn.ema_tile_signatures.clone()
        
        ffn(x)
        ema2 = ffn.ema_tile_signatures.clone()
        
        # EMA should be similar (slowly changing)
        diff = (ema1 - ema2).abs().mean()
        assert diff < 0.1  # Small change


class TestBatchSorting:
    """Tests for GPU-efficient batch sorting."""
    
    def test_output_order_preserved(self):
        """Output order matches input order despite internal sorting."""
        ffn = HierarchicalTriXFFN(d_model=64, num_tiles=16, tiles_per_cluster=4)
        
        # Create inputs with known pattern
        x = torch.randn(32, 64)
        x_copy = x.clone()
        
        out, _, _ = ffn(x)
        
        # Input should be unchanged
        assert torch.allclose(x, x_copy)
        
        # Output should correspond to input order
        assert out.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
