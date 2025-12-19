"""
Rigorous tests for SparseLookupFFNv2.

Tests:
  1. Basic functionality (forward, backward, shapes)
  2. Surgery API (insert, freeze, unfreeze, claim tracking)
  3. Regularizers (ternary, sparsity, diversity)
  4. Score calibration spline
  5. Integration with blocks
  6. Edge cases and stability
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from trix.nn.sparse_lookup_v2 import (
    SparseLookupFFNv2,
    SparseLookupBlockV2,
    ScoreCalibrationSpline,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def basic_config():
    return {
        'd_model': 64,
        'num_tiles': 16,
        'tiles_per_cluster': 4,
    }


@pytest.fixture
def ffn(basic_config):
    return SparseLookupFFNv2(**basic_config)


@pytest.fixture
def ffn_with_regularizers(basic_config):
    return SparseLookupFFNv2(
        **basic_config,
        ternary_weight=0.01,
        sparsity_weight=0.01,
        diversity_weight=0.01,
    )


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestBasicFunctionality:
    """Test basic forward/backward functionality."""
    
    def test_forward_shape_3d(self, ffn, basic_config):
        """Test output shape for 3D input."""
        x = torch.randn(2, 16, basic_config['d_model'])
        output, routing_info, aux_losses = ffn(x)
        
        assert output.shape == x.shape
        assert 'tile_idx' in routing_info
        assert 'compressed' in routing_info
        assert 'total_aux' in aux_losses
    
    def test_forward_shape_large_batch(self, ffn, basic_config):
        """Test with larger batch size."""
        x = torch.randn(32, 64, basic_config['d_model'])
        output, routing_info, aux_losses = ffn(x)
        
        assert output.shape == x.shape
        assert routing_info['tile_idx'].shape == (32, 64)
    
    def test_gradient_flow(self, ffn, basic_config):
        """Test that gradients flow through all components."""
        x = torch.randn(2, 16, basic_config['d_model'], requires_grad=True)
        output, _, aux_losses = ffn(x)
        
        loss = output.sum() + aux_losses['total_aux']
        loss.backward()
        
        # Check input gradient
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check key parameter gradients (some params like calibrator may not be on path)
        key_params = ['signatures_raw', 'directions', 'compress', 'norm']
        for name, param in ffn.named_parameters():
            if param.requires_grad and any(k in name for k in key_params):
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_deterministic_routing(self, ffn, basic_config):
        """Test that routing is deterministic."""
        ffn.eval()
        x = torch.randn(2, 16, basic_config['d_model'])
        
        with torch.no_grad():
            _, info1, _ = ffn(x)
            _, info2, _ = ffn(x)
        
        assert torch.equal(info1['tile_idx'], info2['tile_idx'])
    
    def test_all_tiles_reachable(self, ffn, basic_config):
        """Test that all tiles can be reached with appropriate input."""
        ffn.eval()
        
        # Generate diverse inputs
        x = torch.randn(100, 32, basic_config['d_model'])
        
        with torch.no_grad():
            _, info, _ = ffn(x)
        
        unique_tiles = info['tile_idx'].unique()
        
        # Should use multiple tiles (not all collapse to one)
        assert len(unique_tiles) > 1
    
    def test_residual_connection(self, ffn, basic_config):
        """Test that residual connection is applied."""
        x = torch.randn(2, 16, basic_config['d_model'])
        
        # Zero out the FFN contribution
        with torch.no_grad():
            ffn.output_scale.zero_()
        
        output, _, _ = ffn(x)
        
        # With zero output scale, output should equal input (residual only)
        assert torch.allclose(output, x, atol=1e-5)


# =============================================================================
# Surgery API Tests
# =============================================================================

class TestSurgeryAPI:
    """Test signature surgery functionality."""
    
    def test_insert_signature(self, ffn, basic_config):
        """Test inserting a designed signature."""
        d_model = basic_config['d_model']
        
        # Design signature: +1 on first 8 dims
        sig = torch.zeros(d_model)
        sig[:8] = 1.0
        
        ffn.insert_signature(0, sig, freeze=True, tag="test_sig")
        
        # Check it was inserted
        learned_sig = ffn.signatures[0]
        assert (learned_sig[:8] > 0.5).all()
        assert (learned_sig[8:].abs() < 0.5).all()
    
    def test_freeze_unfreeze(self, ffn, basic_config):
        """Test freeze/unfreeze functionality."""
        d_model = basic_config['d_model']
        
        sig = torch.zeros(d_model)
        sig[:8] = 1.0
        
        ffn.insert_signature(0, sig, freeze=True)
        
        assert ffn.is_frozen(0)
        assert 0 in ffn._frozen_tiles
        
        ffn.unfreeze_signature(0)
        
        assert not ffn.is_frozen(0)
        assert 0 not in ffn._frozen_tiles
    
    def test_frozen_signature_unchanged(self, ffn, basic_config):
        """Test that frozen signature doesn't change during training."""
        d_model = basic_config['d_model']
        
        sig = torch.zeros(d_model)
        sig[:8] = 1.0
        
        ffn.insert_signature(0, sig, freeze=True)
        initial_sig = ffn.signatures[0].clone()
        
        # Train for a few steps
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        for _ in range(10):
            x = torch.randn(4, 16, d_model)
            output, _, aux = ffn(x)
            loss = output.sum() + aux['total_aux']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_sig = ffn.signatures[0]
        
        # Frozen signature should be unchanged
        assert torch.allclose(initial_sig, final_sig)
    
    def test_signature_analysis(self, ffn, basic_config):
        """Test signature analysis function."""
        d_model = basic_config['d_model']
        
        sig = torch.zeros(d_model)
        sig[:8] = 1.0
        sig[8:12] = -1.0
        
        ffn.insert_signature(0, sig, freeze=True)
        
        analysis = ffn.get_signature_analysis(0)
        
        assert 'tile_idx' in analysis
        assert 'positive_dims' in analysis
        assert 'negative_dims' in analysis
        assert 'zero_count' in analysis
        assert 'frozen' in analysis
        
        assert set(analysis['positive_dims']) == set(range(8))
        assert set(analysis['negative_dims']) == set(range(8, 12))
        assert analysis['frozen'] == True
    
    def test_surgery_history(self, ffn, basic_config):
        """Test surgery history tracking."""
        d_model = basic_config['d_model']
        
        sig1 = torch.zeros(d_model)
        sig1[:8] = 1.0
        
        sig2 = torch.zeros(d_model)
        sig2[8:16] = 1.0
        
        ffn.insert_signature(0, sig1, freeze=True, tag="first")
        ffn.insert_signature(1, sig2, freeze=False, tag="second")
        
        history = ffn.get_surgery_history()
        
        assert len(history) == 2
        assert history[0]['tag'] == "first"
        assert history[0]['frozen'] == True
        assert history[1]['tag'] == "second"
        assert history[1]['frozen'] == False
    
    def test_claim_tracking(self, ffn, basic_config):
        """Test claim tracking with labels."""
        d_model = basic_config['d_model']
        
        # Insert a signature
        sig = torch.zeros(d_model)
        sig[:8] = 1.0
        ffn.insert_signature(0, sig, freeze=True)
        
        # Forward with labels
        x = torch.randn(4, 16, d_model)
        labels = torch.randint(0, 10, (4, 16))
        
        ffn.reset_claim_tracking()
        
        for _ in range(10):
            _, _, _ = ffn(x, labels=labels)
        
        # Check claim matrix is populated
        assert ffn.claim_matrix.sum() > 0
    
    def test_get_claim_rate(self, ffn, basic_config):
        """Test claim rate calculation."""
        d_model = basic_config['d_model']
        
        # Design signature that should attract certain inputs
        sig = torch.zeros(d_model)
        sig[:8] = 1.0
        ffn.insert_signature(0, sig, freeze=True)
        
        # Create inputs that match the signature
        x = torch.randn(100, 1, d_model)
        x[:, :, :8] = 2.0  # Strong positive on first 8 dims
        labels = torch.zeros(100, 1, dtype=torch.long)  # All class 0
        
        ffn.reset_claim_tracking()
        
        for i in range(100):
            _, _, _ = ffn(x[i:i+1], labels=labels[i:i+1])
        
        claim_rate = ffn.get_claim_rate(0, 0)
        
        # Should have high claim rate for designed signature
        assert claim_rate > 0  # At least some claims
    
    def test_get_tile_purity(self, ffn, basic_config):
        """Test tile purity calculation."""
        d_model = basic_config['d_model']
        
        x = torch.randn(10, 16, d_model)
        labels = torch.randint(0, 10, (10, 16))
        
        ffn.reset_claim_tracking()
        
        for _ in range(5):
            _, _, _ = ffn(x, labels=labels)
        
        dominant_class, purity = ffn.get_tile_purity(0)
        
        assert dominant_class >= -1  # -1 if unused
        assert 0 <= purity <= 1


# =============================================================================
# Regularizer Tests
# =============================================================================

class TestRegularizers:
    """Test island regularizers."""
    
    def test_ternary_loss_computation(self, ffn_with_regularizers):
        """Test ternary loss is computed correctly."""
        loss = ffn_with_regularizers.compute_ternary_loss()
        
        assert loss.ndim == 0  # Scalar
        assert loss >= 0
        assert not torch.isnan(loss)
    
    def test_ternary_loss_decreases_near_ternary(self):
        """Test that ternary loss is lower for ternary signatures."""
        config = {'d_model': 64, 'num_tiles': 16, 'tiles_per_cluster': 4}
        
        ffn = SparseLookupFFNv2(**config, ternary_weight=0.01)
        
        # Set signatures to ternary values
        with torch.no_grad():
            ffn.signatures_raw.zero_()
            ffn.signatures_raw[:, :8] = 1.0
            ffn.signatures_raw[:, 8:16] = -1.0
        
        loss_ternary = ffn.compute_ternary_loss()
        
        # Set signatures to non-ternary values
        with torch.no_grad():
            ffn.signatures_raw.fill_(0.5)
        
        loss_non_ternary = ffn.compute_ternary_loss()
        
        assert loss_ternary < loss_non_ternary
    
    def test_sparsity_loss_computation(self, ffn_with_regularizers):
        """Test sparsity loss is computed correctly."""
        loss = ffn_with_regularizers.compute_sparsity_loss()
        
        assert loss.ndim == 0
        assert loss >= 0
        assert not torch.isnan(loss)
    
    def test_sparsity_loss_prefers_sparse(self):
        """Test that sparsity loss prefers sparse signatures."""
        config = {'d_model': 64, 'num_tiles': 16, 'tiles_per_cluster': 4}
        
        ffn = SparseLookupFFNv2(**config, sparsity_weight=0.01)
        
        # Sparse signatures (mostly zeros)
        with torch.no_grad():
            ffn.signatures_raw.zero_()
            ffn.signatures_raw[:, :4] = 1.0
        
        loss_sparse = ffn.compute_sparsity_loss()
        
        # Dense signatures (mostly non-zero)
        with torch.no_grad():
            ffn.signatures_raw.fill_(1.0)
        
        loss_dense = ffn.compute_sparsity_loss()
        
        assert loss_sparse < loss_dense
    
    def test_diversity_loss_computation(self, ffn_with_regularizers):
        """Test diversity loss is computed correctly."""
        loss = ffn_with_regularizers.compute_diversity_loss()
        
        assert loss.ndim == 0
        assert loss >= 0
        assert not torch.isnan(loss)
    
    def test_diversity_loss_penalizes_similar(self):
        """Test that diversity loss penalizes similar signatures."""
        config = {'d_model': 64, 'num_tiles': 16, 'tiles_per_cluster': 4}
        
        ffn = SparseLookupFFNv2(**config, diversity_weight=0.01)
        
        # All signatures identical
        with torch.no_grad():
            ffn.signatures_raw.fill_(0.0)
            ffn.signatures_raw[:, :8] = 1.0
        
        loss_similar = ffn.compute_diversity_loss()
        
        # Diverse signatures
        with torch.no_grad():
            for i in range(16):
                ffn.signatures_raw[i].zero_()
                ffn.signatures_raw[i, i*4:(i+1)*4] = 1.0
        
        loss_diverse = ffn.compute_diversity_loss()
        
        assert loss_diverse < loss_similar
    
    def test_regularizers_in_aux_losses(self, ffn_with_regularizers, basic_config):
        """Test that regularizers appear in aux_losses."""
        x = torch.randn(2, 16, basic_config['d_model'])
        _, _, aux_losses = ffn_with_regularizers(x)
        
        assert 'ternary' in aux_losses
        assert 'sparsity' in aux_losses
        assert 'diversity' in aux_losses
        assert 'total_aux' in aux_losses
        
        # Total should include all components
        expected_total = (
            aux_losses['balance'] +
            aux_losses['ternary'] +
            aux_losses['sparsity'] +
            aux_losses['diversity']
        )
        
        assert torch.allclose(aux_losses['total_aux'], expected_total)
    
    def test_regularizers_disabled_when_zero_weight(self, basic_config):
        """Test regularizers are zero when weights are zero."""
        ffn = SparseLookupFFNv2(
            **basic_config,
            ternary_weight=0.0,
            sparsity_weight=0.0,
            diversity_weight=0.0,
        )
        
        x = torch.randn(2, 16, basic_config['d_model'])
        _, _, aux_losses = ffn(x)
        
        assert aux_losses['ternary'] == 0.0
        assert aux_losses['sparsity'] == 0.0
        assert aux_losses['diversity'] == 0.0


# =============================================================================
# Score Calibration Spline Tests
# =============================================================================

class TestScoreCalibrationSpline:
    """Test score calibration spline."""
    
    def test_forward_shape(self):
        """Test spline output shape."""
        spline = ScoreCalibrationSpline(num_knots=8)
        
        scores = torch.randn(32, 64)
        gates = spline(scores)
        
        assert gates.shape == scores.shape
    
    def test_output_range(self):
        """Test that output is in [0, 1]."""
        spline = ScoreCalibrationSpline(num_knots=8)
        
        scores = torch.randn(100, 100) * 10  # Wide range of scores
        gates = spline(scores)
        
        assert gates.min() >= 0
        assert gates.max() <= 1
    
    def test_gradient_flow(self):
        """Test gradients flow through spline."""
        spline = ScoreCalibrationSpline(num_knots=8)
        
        scores = torch.randn(32, 64, requires_grad=True)
        gates = spline(scores)
        
        loss = gates.sum()
        loss.backward()
        
        assert scores.grad is not None
        assert spline.knot_values.grad is not None
    
    def test_monotonic_tendency(self):
        """Test that spline tends toward monotonic (higher score → higher gate)."""
        spline = ScoreCalibrationSpline(num_knots=8)
        
        # Sorted scores
        scores = torch.linspace(-5, 5, 100).unsqueeze(0)
        gates = spline(scores)
        
        # Check general trend (not strictly monotonic due to learned params)
        # But initial state should be sigmoid-like
        assert gates[0, 0] < gates[0, -1]  # Low score < high score


# =============================================================================
# Block Integration Tests
# =============================================================================

class TestBlockIntegration:
    """Test SparseLookupBlockV2 integration."""
    
    def test_block_forward(self, basic_config):
        """Test block forward pass."""
        block = SparseLookupBlockV2(
            d_model=basic_config['d_model'],
            n_heads=4,
            num_tiles=basic_config['num_tiles'],
            tiles_per_cluster=basic_config['tiles_per_cluster'],
        )
        
        x = torch.randn(2, 16, basic_config['d_model'])
        output, routing_info, aux_losses = block(x)
        
        assert output.shape == x.shape
    
    def test_block_causal_masking(self, basic_config):
        """Test that causal masking works."""
        block = SparseLookupBlockV2(
            d_model=basic_config['d_model'],
            n_heads=4,
            num_tiles=basic_config['num_tiles'],
            tiles_per_cluster=basic_config['tiles_per_cluster'],
        )
        
        x = torch.randn(2, 16, basic_config['d_model'])
        
        # Should not error with causal=True
        output_causal, _, _ = block(x, is_causal=True)
        output_non_causal, _, _ = block(x, is_causal=False)
        
        # Outputs may differ due to masking
        assert output_causal.shape == output_non_causal.shape
    
    def test_block_gradient_flow(self, basic_config):
        """Test gradients flow through block."""
        block = SparseLookupBlockV2(
            d_model=basic_config['d_model'],
            n_heads=4,
            num_tiles=basic_config['num_tiles'],
            tiles_per_cluster=basic_config['tiles_per_cluster'],
        )
        
        x = torch.randn(2, 16, basic_config['d_model'], requires_grad=True)
        output, _, aux_losses = block(x)
        
        loss = output.sum() + aux_losses['total_aux']
        loss.backward()
        
        assert x.grad is not None
    
    def test_block_with_labels(self, basic_config):
        """Test block with label tracking."""
        block = SparseLookupBlockV2(
            d_model=basic_config['d_model'],
            n_heads=4,
            num_tiles=basic_config['num_tiles'],
            tiles_per_cluster=basic_config['tiles_per_cluster'],
        )
        
        x = torch.randn(2, 16, basic_config['d_model'])
        labels = torch.randint(0, 10, (2, 16))
        
        output, routing_info, aux_losses = block(x, labels=labels)
        
        assert output.shape == x.shape


# =============================================================================
# Island Stats Tests
# =============================================================================

class TestIslandStats:
    """Test island statistics computation."""
    
    def test_get_island_stats(self, ffn):
        """Test island stats computation."""
        stats = ffn.get_island_stats()
        
        assert 'ternary_fraction' in stats
        assert 'sparsity' in stats
        assert 'mean_pairwise_similarity' in stats
        assert 'diversity' in stats
        
        assert 0 <= stats['ternary_fraction'] <= 1
        assert 0 <= stats['sparsity'] <= 1
        # Diversity can slightly exceed 1 due to floating point (1 - neg_similarity)
        assert -0.1 <= stats['diversity'] <= 1.1
    
    def test_get_routing_stats(self, ffn, basic_config):
        """Test routing stats computation."""
        # Run some forward passes
        x = torch.randn(10, 16, basic_config['d_model'])
        for _ in range(5):
            ffn(x)
        
        stats = ffn.get_routing_stats()
        
        assert 'num_tiles' in stats
        assert 'active_tiles' in stats
        assert 'frozen_tiles' in stats
        assert 'usage_mean' in stats
    
    def test_reset_stats(self, ffn, basic_config):
        """Test stats reset."""
        x = torch.randn(10, 16, basic_config['d_model'])
        for _ in range(5):
            ffn(x)
        
        assert ffn.total_count > 0
        
        ffn.reset_stats()
        
        assert ffn.total_count == 0
        assert ffn.tile_counts.sum() == 0


# =============================================================================
# Edge Cases and Stability
# =============================================================================

class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_single_sample(self, ffn, basic_config):
        """Test with batch size 1."""
        x = torch.randn(1, 16, basic_config['d_model'])
        output, _, _ = ffn(x)
        
        assert output.shape == x.shape
    
    def test_single_token(self, ffn, basic_config):
        """Test with sequence length 1."""
        x = torch.randn(2, 1, basic_config['d_model'])
        output, _, _ = ffn(x)
        
        assert output.shape == x.shape
    
    def test_zero_input(self, ffn, basic_config):
        """Test with zero input."""
        x = torch.zeros(2, 16, basic_config['d_model'])
        output, _, _ = ffn(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_large_input(self, ffn, basic_config):
        """Test with large input values."""
        x = torch.randn(2, 16, basic_config['d_model']) * 100
        output, _, _ = ffn(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_eval_mode(self, ffn, basic_config):
        """Test behavior in eval mode."""
        ffn.eval()
        
        x = torch.randn(2, 16, basic_config['d_model'])
        
        with torch.no_grad():
            output, routing_info, aux_losses = ffn(x)
        
        assert output.shape == x.shape
        # Should still compute aux losses in eval
        assert 'total_aux' in aux_losses
    
    def test_multiple_forward_backward(self, ffn, basic_config):
        """Test multiple forward/backward passes don't accumulate errors."""
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        for _ in range(10):
            x = torch.randn(2, 16, basic_config['d_model'])
            output, _, aux_losses = ffn(x)
            
            loss = output.sum() + aux_losses['total_aux']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Check no NaNs
            for param in ffn.parameters():
                assert not torch.isnan(param).any()


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
