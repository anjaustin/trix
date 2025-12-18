"""
TriX QAT Tests

Tests for quantization-aware training components.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn

from trix.qat import (
    TernaryQuantizer,
    SoftTernaryQuantizer,
    Top1Gate,
    TriXLinearQAT,
    progressive_quantization_schedule,
    QATTrainer,
)


class TestTernaryQuantizer:
    """Tests for hard ternary quantization."""
    
    def test_quantization_values(self):
        """Should produce only {-1, 0, +1}."""
        weights = torch.randn(100, 100)
        quantized = TernaryQuantizer.apply(weights, 0.05)
        
        unique = quantized.unique()
        assert len(unique) <= 3
        for val in unique:
            assert val.item() in [-1.0, 0.0, 1.0]
    
    def test_threshold(self):
        """Values below threshold should become zero."""
        weights = torch.tensor([0.01, -0.01, 0.1, -0.1])
        quantized = TernaryQuantizer.apply(weights, 0.05)
        
        assert quantized[0].item() == 0.0  # Below threshold
        assert quantized[1].item() == 0.0  # Below threshold
        assert quantized[2].item() == 1.0  # Above threshold
        assert quantized[3].item() == -1.0  # Below -threshold
    
    def test_gradient_flow(self):
        """Gradients should flow via STE."""
        weights = torch.randn(10, 10, requires_grad=True)
        quantized = TernaryQuantizer.apply(weights, 0.05)
        loss = quantized.sum()
        loss.backward()
        
        assert weights.grad is not None
        # Gradient should be non-zero for weights in valid range
        assert weights.grad.abs().sum() > 0


class TestSoftTernaryQuantizer:
    """Tests for soft ternary quantization."""
    
    def test_output_range(self):
        """Output should be in [-1, 1]."""
        soft_quant = SoftTernaryQuantizer(initial_temp=5.0)
        weights = torch.randn(100, 100) * 2
        output = soft_quant(weights)
        
        assert output.min() >= -1.0
        assert output.max() <= 1.0
    
    def test_temperature_effect(self):
        """Higher temperature should give sharper quantization."""
        weights = torch.randn(100, 100)
        
        soft_low = SoftTernaryQuantizer(initial_temp=1.0)
        soft_high = SoftTernaryQuantizer(initial_temp=10.0)
        
        out_low = soft_low(weights)
        out_high = soft_high(weights)
        
        # Higher temp should have more values near ±1
        near_one_low = ((out_low.abs() > 0.9).float().mean())
        near_one_high = ((out_high.abs() > 0.9).float().mean())
        
        assert near_one_high > near_one_low


class TestTop1Gate:
    """Tests for hard top-1 gating."""
    
    def test_one_hot_output(self):
        """Output should be one-hot."""
        logits = torch.randn(8, 4)
        gate = Top1Gate.apply(logits)
        
        # Each row should sum to 1
        assert torch.allclose(gate.sum(dim=1), torch.ones(8))
        
        # Each row should have exactly one 1
        assert torch.all(gate.max(dim=1).values == 1.0)
    
    def test_argmax_position(self):
        """One should be at argmax position."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        gate = Top1Gate.apply(logits)
        
        assert gate[0, 3].item() == 1.0
        assert gate[0, :3].sum().item() == 0.0
    
    def test_gradient_flow(self):
        """Gradients should pass through."""
        logits = torch.randn(4, 4, requires_grad=True)
        gate = Top1Gate.apply(logits)
        loss = gate.sum()
        loss.backward()
        
        assert logits.grad is not None


class TestTriXLinearQAT:
    """Tests for QAT linear layer."""
    
    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        layer = TriXLinearQAT(64, 128, num_tiles=4)
        x = torch.randn(8, 64)
        out = layer(x)
        
        assert out.shape == (8, 128)
    
    def test_quant_modes(self):
        """All quantization modes should work."""
        for mode in ['none', 'soft', 'ste', 'progressive']:
            layer = TriXLinearQAT(32, 64, quant_mode=mode)
            x = torch.randn(4, 32)
            out = layer(x)
            assert out.shape == (4, 64)
    
    def test_gradient_flow(self):
        """Gradients should flow through."""
        layer = TriXLinearQAT(32, 64, quant_mode='ste')
        x = torch.randn(4, 32)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        # Weight gradients might be zero due to STE, but scales should have gradient
        assert layer.scales.grad is not None
    
    def test_sparsity_metric(self):
        """Sparsity calculation should work."""
        layer = TriXLinearQAT(32, 64, quant_mode='ste', threshold=0.5)
        # Most weights should be non-zero with default init
        sparsity = layer.get_sparsity()
        assert 0.0 <= sparsity <= 1.0
    
    def test_ternary_distribution(self):
        """Ternary distribution should sum to 1."""
        layer = TriXLinearQAT(32, 64, quant_mode='ste')
        dist = layer.get_ternary_distribution()
        
        total = dist['neg'] + dist['zero'] + dist['pos']
        assert abs(total - 1.0) < 0.01


class TestProgressiveSchedule:
    """Tests for progressive quantization schedule."""
    
    def test_start_end_values(self):
        """Should start at start_temp and end at end_temp."""
        temp_0 = progressive_quantization_schedule(0, 100, 1.0, 10.0)
        temp_100 = progressive_quantization_schedule(100, 100, 1.0, 10.0)
        
        assert abs(temp_0 - 1.0) < 0.01
        assert abs(temp_100 - 10.0) < 0.01
    
    def test_monotonic_increase(self):
        """Temperature should increase monotonically."""
        temps = [progressive_quantization_schedule(e, 100, 1.0, 10.0) for e in range(101)]
        
        for i in range(len(temps) - 1):
            assert temps[i] <= temps[i + 1]


class TestQATTrainer:
    """Tests for QAT trainer helper."""
    
    def test_epoch_step(self):
        """step_epoch should update temperature."""
        model = nn.Sequential(
            TriXLinearQAT(32, 64, quant_mode='progressive'),
            TriXLinearQAT(64, 32, quant_mode='progressive'),
        )
        
        trainer = QATTrainer(model, total_epochs=10)
        
        initial_temp = model[0].soft_quant.temperature.item()
        trainer.step_epoch()
        new_temp = model[0].soft_quant.temperature.item()
        
        assert new_temp > initial_temp
    
    def test_model_sparsity(self):
        """get_model_sparsity should return average."""
        model = nn.Sequential(
            TriXLinearQAT(32, 64, quant_mode='ste'),
            TriXLinearQAT(64, 32, quant_mode='ste'),
        )
        
        trainer = QATTrainer(model, total_epochs=10)
        sparsity = trainer.get_model_sparsity()
        
        assert 0.0 <= sparsity <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
