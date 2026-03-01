"""
Integration Tests for TriX

Purpose:
    Validate that the recommended TriX architecture (SparseLookupFFN) trains
    end-to-end without catastrophic failures. These tests catch issues that
    unit tests miss: training instability, routing collapse, numerical problems
    that emerge over multiple steps.

What these tests catch:
    - NaN/Inf during training (numerical instability)
    - Loss not decreasing (broken gradients, bad initialization)
    - Routing collapse (all tokens going to same tile)
    - Multi-layer interaction bugs

What these tests do NOT catch:
    - Absolute performance thresholds (that's the benchmark's job)
    - Real dataset convergence (requires data download)
    - Comparison between FFN variants
    - Subtle quality regressions

Design decisions:
    - Uses synthetic data (no network dependencies, deterministic)
    - 50 training steps (enough to surface issues, fast enough for CI)
    - Tests SparseLookupBlock stack (attention + FFN) as recommended architecture
    - Conservative thresholds to avoid flaky failures

Runtime target: <15 seconds on CPU
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from trix import SparseLookupFFN, SparseLookupBlock


# =============================================================================
# Configuration Constants
# =============================================================================

# Number of training steps to run.
# Rationale: 50 steps is enough to surface gradient issues and routing collapse,
# but fast enough to run in CI (<15 seconds on CPU).
TRAIN_STEPS = 50

# Required loss improvement ratio (final_loss < initial_loss * threshold).
# Rationale: Even on synthetic data, a working model should improve by at least
# 20% in 50 steps. This is conservative to avoid flaky tests.
LOSS_IMPROVEMENT_THRESHOLD = 0.8

# Minimum fraction of tiles that should be used during training.
# Rationale: With 16 tiles, we expect diverse routing. If utilization drops
# below 0.3 (only ~5 tiles used), routing may be collapsing.
MIN_TILE_UTILIZATION = 0.3

# Model dimensions (small for speed, large enough to be meaningful)
D_MODEL = 64
N_HEADS = 4
NUM_TILES = 16
N_LAYERS = 2

# Training hyperparameters
BATCH_SIZE = 8
SEQ_LEN = 32
LEARNING_RATE = 1e-3

# Random seed for reproducibility
SEED = 42


# =============================================================================
# Helper Functions
# =============================================================================

def _build_mini_model() -> nn.Module:
    """
    Build a minimal multi-layer SparseLookupBlock stack.
    
    Architecture:
        Input -> Block1 -> Block2 -> Output projection
        
    Each block contains:
        - LayerNorm + Multi-head Attention + Residual
        - LayerNorm + SparseLookupFFN + Residual
    
    Returns:
        nn.Module with forward(x) -> (logits, aux_loss)
    """
    class MiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(D_MODEL, D_MODEL)  # Identity-like init
            self.blocks = nn.ModuleList([
                SparseLookupBlock(
                    d_model=D_MODEL,
                    n_heads=N_HEADS,
                    num_tiles=NUM_TILES,
                )
                for _ in range(N_LAYERS)
            ])
            self.output_proj = nn.Linear(D_MODEL, D_MODEL)
        
        def forward(self, x: torch.Tensor) -> tuple:
            """
            Args:
                x: Input tensor [batch, seq, d_model]
            
            Returns:
                output: Transformed tensor [batch, seq, d_model]
                total_aux: Combined auxiliary loss from all blocks
            """
            x = self.embed(x)
            total_aux = torch.tensor(0.0, device=x.device)
            
            for block in self.blocks:
                x, _info, aux = block(x, is_causal=True)
                total_aux = total_aux + aux.get('total_aux', 0.0)
            
            return self.output_proj(x), total_aux
        
        def get_routing_stats(self) -> dict:
            """Aggregate routing stats from all FFN blocks."""
            all_active = []
            all_usage = []
            
            for block in self.blocks:
                block.ffn.reset_stats()
            
            return {'blocks': len(self.blocks)}
    
    return MiniModel()


def _generate_synthetic_data(
    batch_size: int,
    seq_len: int,
    d_model: int,
    seed: int,
) -> tuple:
    """
    Generate synthetic training data with a learnable pattern.
    
    Task: Predict a shifted/transformed version of the input.
    This is simple enough to learn quickly but requires actual computation.
    
    Args:
        batch_size: Number of sequences
        seq_len: Sequence length
        d_model: Model dimension
        seed: Random seed for reproducibility
    
    Returns:
        x: Input tensor [batch, seq, d_model]
        target: Target tensor [batch, seq, d_model]
    """
    torch.manual_seed(seed)
    
    # Input: random features
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Target: transformed version (shift + scale + noise)
    # This creates a learnable pattern without being trivial
    target = x * 0.8 + torch.roll(x, shifts=1, dims=1) * 0.2
    target = target + torch.randn_like(target) * 0.1
    
    return x, target


def _check_finite(tensor: torch.Tensor, name: str) -> None:
    """Assert tensor contains no NaN or Inf values."""
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf"


def _get_tile_utilization(model: nn.Module, x: torch.Tensor) -> float:
    """
    Measure what fraction of tiles are being used.
    
    Returns:
        Fraction of tiles that received at least one token (0.0 to 1.0)
    """
    model.eval()
    all_tiles = set()
    
    with torch.no_grad():
        h = model.embed(x)
        for block in model.blocks:
            h, info, _ = block(h, is_causal=True)
            if 'tile_idx' in info:
                tiles = info['tile_idx'].flatten().tolist()
                all_tiles.update(tiles)
    
    model.train()
    return len(all_tiles) / NUM_TILES


# =============================================================================
# Tests
# =============================================================================

class TestSmokeIntegration:
    """
    Smoke tests that validate training works end-to-end.
    
    These are not performance tests - they verify the system doesn't
    break catastrophically during training.
    """
    
    def test_training_converges(self):
        """
        Train for TRAIN_STEPS and verify:
        1. No NaN/Inf in outputs
        2. Loss decreases by at least (1 - LOSS_IMPROVEMENT_THRESHOLD)
        3. Routing uses at least MIN_TILE_UTILIZATION of tiles
        """
        torch.manual_seed(SEED)
        
        # Setup
        model = _build_mini_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        x, target = _generate_synthetic_data(BATCH_SIZE, SEQ_LEN, D_MODEL, SEED)
        
        # Record initial loss
        model.train()
        with torch.no_grad():
            output, aux_loss = model(x)
            initial_loss = F.mse_loss(output, target) + aux_loss * 0.01
            initial_loss_val = initial_loss.item()
        
        _check_finite(output, "initial output")
        
        # Training loop
        losses = []
        for step in range(TRAIN_STEPS):
            optimizer.zero_grad()
            output, aux_loss = model(x)
            loss = F.mse_loss(output, target) + aux_loss * 0.01
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            # Check for NaN/Inf every 10 steps
            if step % 10 == 0:
                _check_finite(output, f"output at step {step}")
                _check_finite(loss, f"loss at step {step}")
        
        final_loss = losses[-1]
        
        # Assertion 1: Final output is finite
        with torch.no_grad():
            final_output, _ = model(x)
        _check_finite(final_output, "final output")
        
        # Assertion 2: Loss improved
        assert final_loss < initial_loss_val * LOSS_IMPROVEMENT_THRESHOLD, (
            f"Loss did not improve enough: {initial_loss_val:.4f} -> {final_loss:.4f} "
            f"(required {LOSS_IMPROVEMENT_THRESHOLD * 100:.0f}% of initial)"
        )
        
        # Assertion 3: Routing is not collapsed
        utilization = _get_tile_utilization(model, x)
        assert utilization >= MIN_TILE_UTILIZATION, (
            f"Tile utilization too low: {utilization:.2f} < {MIN_TILE_UTILIZATION} "
            f"(possible routing collapse)"
        )
    
    def test_eval_mode_stable(self):
        """
        Verify eval mode produces finite outputs.
        
        Some bugs only manifest in eval mode (e.g., uninitialized buffers,
        different code paths for dropout/batchnorm).
        """
        torch.manual_seed(SEED)
        
        model = _build_mini_model()
        model.eval()
        
        x, _ = _generate_synthetic_data(BATCH_SIZE, SEQ_LEN, D_MODEL, SEED)
        
        with torch.no_grad():
            output, aux_loss = model(x)
        
        _check_finite(output, "eval output")
        _check_finite(aux_loss, "eval aux_loss")
        
        # Output shape should match input
        assert output.shape == x.shape, (
            f"Output shape {output.shape} != input shape {x.shape}"
        )
    
    def test_gradient_flow_through_stack(self):
        """
        Verify gradients flow through all layers of the stack.
        
        A common bug is gradients being blocked somewhere in the middle,
        causing early layers to not learn.
        """
        torch.manual_seed(SEED)
        
        model = _build_mini_model()
        x, target = _generate_synthetic_data(BATCH_SIZE, SEQ_LEN, D_MODEL, SEED)
        
        output, aux_loss = model(x)
        loss = F.mse_loss(output, target) + aux_loss * 0.01
        loss.backward()
        
        # Check gradients exist for key parameters in each layer
        # Embedding layer
        assert model.embed.weight.grad is not None, "No gradient for embed layer"
        assert model.embed.weight.grad.abs().sum() > 0, "Zero gradient for embed layer"
        
        # Each block's FFN
        for i, block in enumerate(model.blocks):
            assert block.ffn.directions.grad is not None, (
                f"No gradient for block {i} FFN directions"
            )
            assert block.ffn.directions.grad.abs().sum() > 0, (
                f"Zero gradient for block {i} FFN directions"
            )
        
        # Output projection
        assert model.output_proj.weight.grad is not None, "No gradient for output_proj"
        assert model.output_proj.weight.grad.abs().sum() > 0, "Zero gradient for output_proj"


class TestSparseLookupFFNSmoke:
    """
    Direct smoke tests for SparseLookupFFN (the recommended FFN).
    
    Tests the FFN in isolation to catch FFN-specific issues.
    """
    
    def test_ffn_training_smoke(self):
        """
        SparseLookupFFN trains for multiple steps without numerical issues.
        
        This test validates stability, not convergence. The full model test
        (TestSmokeIntegration.test_training_converges) validates convergence.
        
        We check:
        - No NaN/Inf during training
        - Gradients are computed
        - aux_losses are reasonable
        """
        torch.manual_seed(SEED)
        
        model = SparseLookupFFN(d_model=D_MODEL, num_tiles=NUM_TILES)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        target = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        
        for step in range(TRAIN_STEPS):
            optimizer.zero_grad()
            output, info, aux = model(x)
            loss = F.mse_loss(output, target) + aux['total_aux'] * 0.01
            loss.backward()
            optimizer.step()
            
            # Check numerical stability
            _check_finite(output, f"output at step {step}")
            _check_finite(loss, f"loss at step {step}")
            
            # Check aux_losses are reasonable (not exploding)
            assert aux['total_aux'].item() < 100, (
                f"aux_loss exploded at step {step}: {aux['total_aux'].item()}"
            )
        
        # Verify gradients flowed to key parameters
        optimizer.zero_grad()
        output, _, aux = model(x)
        loss = F.mse_loss(output, target) + aux['total_aux'] * 0.01
        loss.backward()
        
        assert model.directions.grad is not None, "No gradient for directions"
        assert model.directions.grad.abs().sum() > 0, "Zero gradient for directions"
    
    def test_ffn_routing_diversity(self):
        """SparseLookupFFN uses multiple tiles, not just one."""
        torch.manual_seed(SEED)
        
        model = SparseLookupFFN(d_model=D_MODEL, num_tiles=NUM_TILES)
        model.eval()
        
        # Use diverse inputs to encourage diverse routing
        x = torch.randn(32, 64, D_MODEL)  # Larger batch for diversity
        
        with torch.no_grad():
            _, info, _ = model(x)
        
        unique_tiles = info['tile_idx'].unique()
        utilization = len(unique_tiles) / NUM_TILES
        
        assert utilization >= MIN_TILE_UTILIZATION, (
            f"Only {len(unique_tiles)}/{NUM_TILES} tiles used ({utilization:.1%})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
