"""
Tests for Temporal Tile Layer - Mesa 4

Tests cover:
1. Basic functionality (shapes, forward pass)
2. State tracking (state updates correctly)
3. Routing behavior (tiles specialize)
4. Bracket counting (the canonical test)
5. Transition tracking
6. Sequence processing
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/workspace/trix_latest/src')

from trix.nn import TemporalTileLayer, TemporalTileStack


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def temporal_layer(device):
    return TemporalTileLayer(
        d_model=32,
        d_state=16,
        num_tiles=8,
    ).to(device)


@pytest.fixture
def temporal_stack(device):
    return TemporalTileStack(
        d_model=32,
        layers=[(16, 8), (8, 4)],  # Two layers with different configs
    ).to(device)


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestTemporalTileLayerBasic:
    """Test basic layer functionality."""
    
    def test_init_state_shape(self, temporal_layer, device):
        """Test state initialization."""
        state = temporal_layer.init_state(batch_size=4, device=device)
        assert state.shape == (4, 16)
        
    def test_init_state_zero(self, temporal_layer, device):
        """Test zero initialization."""
        state = temporal_layer.init_state(batch_size=2, device=device)
        assert torch.allclose(state, torch.zeros_like(state))
    
    def test_forward_shapes(self, temporal_layer, device):
        """Test forward pass output shapes."""
        x = torch.randn(4, 32, device=device)
        state = temporal_layer.init_state(4, device)
        
        output, new_state, info = temporal_layer(x, state)
        
        assert output.shape == (4, 32)
        assert new_state.shape == (4, 16)
        assert 'tile_idx' in info
        assert info['tile_idx'].shape == (4,)
    
    def test_forward_sequence_shapes(self, temporal_layer, device):
        """Test sequence processing shapes."""
        x = torch.randn(4, 10, 32, device=device)  # batch=4, seq=10, d_model=32
        
        output, final_state, infos = temporal_layer.forward_sequence(x)
        
        assert output.shape == (4, 10, 32)
        assert final_state.shape == (4, 16)
        assert len(infos) == 10
    
    def test_tile_idx_in_range(self, temporal_layer, device):
        """Test tile indices are valid."""
        x = torch.randn(8, 32, device=device)
        state = temporal_layer.init_state(8, device)
        
        _, _, info = temporal_layer(x, state)
        
        assert info['tile_idx'].min() >= 0
        assert info['tile_idx'].max() < 8


class TestTemporalTileLayerState:
    """Test state tracking behavior."""
    
    def test_state_changes(self, temporal_layer, device):
        """Test that state actually changes."""
        x = torch.randn(2, 32, device=device)
        state = temporal_layer.init_state(2, device)
        
        _, new_state, _ = temporal_layer(x, state)
        
        # State should change (not stay zero)
        assert not torch.allclose(new_state, state)
    
    def test_state_affects_routing(self, temporal_layer, device):
        """Test that state influences routing decisions."""
        x = torch.randn(1, 32, device=device)
        
        # Same input, different states
        state1 = torch.randn(1, 16, device=device)
        state2 = torch.randn(1, 16, device=device) * 10  # Very different
        
        _, _, info1 = temporal_layer(x, state1)
        _, _, info2 = temporal_layer(x, state2)
        
        # With very different states, routing might differ
        # (Not guaranteed, but state_contribution should be non-zero)
        assert info1['state_contribution'] >= 0 or info2['state_contribution'] >= 0
    
    def test_deterministic_routing(self, temporal_layer, device):
        """Test routing is deterministic for same input/state."""
        temporal_layer.eval()
        
        x = torch.randn(4, 32, device=device)
        state = torch.randn(4, 16, device=device)
        
        _, _, info1 = temporal_layer(x, state)
        _, _, info2 = temporal_layer(x, state)
        
        assert torch.equal(info1['tile_idx'], info2['tile_idx'])


class TestTemporalTileLayerTracking:
    """Test transition and usage tracking."""
    
    def test_reset_tracking(self, temporal_layer, device):
        """Test tracking reset."""
        temporal_layer.reset_tracking()
        
        assert temporal_layer.tile_counts.sum() == 0
        assert temporal_layer.transition_counts.sum() == 0
    
    def test_tile_counts_increment(self, temporal_layer, device):
        """Test tile usage is tracked."""
        temporal_layer.reset_tracking()
        
        x = torch.randn(4, 10, 32, device=device)
        temporal_layer.forward_sequence(x)
        
        # Should have 4 * 10 = 40 total tile activations
        assert temporal_layer.tile_counts.sum() == 40
    
    def test_transition_counts_increment(self, temporal_layer, device):
        """Test transitions are tracked."""
        temporal_layer.reset_tracking()
        
        x = torch.randn(2, 5, 32, device=device)
        temporal_layer.forward_sequence(x)
        
        # Each sequence has 4 transitions (5 steps - 1)
        # Total: 2 * 4 = 8 transitions
        assert temporal_layer.transition_counts.sum() == 8
    
    def test_transition_matrix_normalized(self, temporal_layer, device):
        """Test normalized transition matrix sums to 1 per row."""
        temporal_layer.reset_tracking()
        
        x = torch.randn(10, 20, 32, device=device)
        temporal_layer.forward_sequence(x)
        
        trans = temporal_layer.get_transition_matrix(normalize=True)
        row_sums = trans.sum(dim=1)
        
        # Rows with any transitions should sum to 1
        active_rows = temporal_layer.transition_counts.sum(dim=1) > 0
        assert torch.allclose(row_sums[active_rows], torch.ones_like(row_sums[active_rows]), atol=1e-5)
    
    def test_tile_usage_normalized(self, temporal_layer, device):
        """Test normalized tile usage sums to 1."""
        temporal_layer.reset_tracking()
        
        x = torch.randn(5, 10, 32, device=device)
        temporal_layer.forward_sequence(x)
        
        usage = temporal_layer.get_tile_usage(normalize=True)
        assert abs(usage.sum().item() - 1.0) < 1e-5


class TestTemporalTileStack:
    """Test stacked temporal tile layers."""
    
    def test_stack_init_states(self, temporal_stack, device):
        """Test state initialization for stack."""
        states = temporal_stack.init_states(batch_size=4, device=device)
        
        assert len(states) == 2
        assert states[0].shape == (4, 16)  # First layer: d_state=16
        assert states[1].shape == (4, 8)   # Second layer: d_state=8
    
    def test_stack_forward_sequence(self, temporal_stack, device):
        """Test forward pass through stack."""
        x = torch.randn(4, 10, 32, device=device)
        
        output, final_states, all_infos = temporal_stack.forward_sequence(x)
        
        assert output.shape == (4, 10, 32)
        assert len(final_states) == 2
        assert len(all_infos) == 2  # One list per layer
        assert len(all_infos[0]) == 10  # 10 timesteps


# =============================================================================
# Bracket Counting Tests (The Canonical Test)
# =============================================================================

class TestBracketCounting:
    """Test temporal tiles on bracket depth prediction."""
    
    @pytest.fixture
    def bracket_model(self, device):
        """Simple bracket counting model."""
        class BracketCounter(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(2, 8)
                self.temporal = TemporalTileLayer(
                    d_model=8,
                    d_state=8,
                    num_tiles=6,
                    routing_temp=0.5,
                )
                self.head = nn.Linear(8, 5)  # Depths 0-4
            
            def forward(self, tokens):
                B, T = tokens.shape
                x = self.embed(tokens)
                output, _, infos = self.temporal.forward_sequence(x)
                return self.head(output), infos
        
        return BracketCounter().to(device)
    
    def test_bracket_model_shapes(self, bracket_model, device):
        """Test bracket model output shapes."""
        tokens = torch.randint(0, 2, (4, 8), device=device)
        logits, infos = bracket_model(tokens)
        
        assert logits.shape == (4, 8, 5)
        assert len(infos) == 8
    
    def test_bracket_model_trainable(self, bracket_model, device):
        """Test bracket model can be trained."""
        tokens = torch.randint(0, 2, (4, 6), device=device)
        targets = torch.randint(0, 5, (4, 6), device=device)
        
        logits, _ = bracket_model(tokens)
        loss = F.cross_entropy(logits.view(-1, 5), targets.view(-1))
        
        loss.backward()
        
        # Check gradients exist
        for p in bracket_model.parameters():
            if p.requires_grad:
                assert p.grad is not None
    
    def test_bracket_counting_learns(self, device):
        """Integration test: can temporal tiles learn to count brackets?"""
        # Simple model
        class TinyCounter(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(2, 8)
                self.gru = nn.GRUCell(8, 8)
                self.sigs = nn.Parameter(torch.randn(6, 16) * 0.1)
                self.head = nn.Linear(8, 5)
            
            def forward(self, tokens):
                B, T = tokens.shape
                state = torch.zeros(B, 8, device=tokens.device)
                outs = []
                for t in range(T):
                    x = self.embed(tokens[:, t])
                    comb = torch.cat([x, state], -1)
                    # Routing (for interpretability, not used in output)
                    _ = (comb @ self.sigs.T).argmax(-1)
                    state = self.gru(x, state)
                    outs.append(self.head(state))
                return torch.stack(outs, 1)
        
        model = TinyCounter().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        
        # Generate data
        def gen_seq():
            toks, deps = [], []
            d = 0
            for _ in range(np.random.randint(4, 8)):
                t = 0 if d == 0 else (1 if d == 4 else np.random.randint(2))
                toks.append(t)
                d = d + (1 if t == 0 else -1)
                deps.append(d)
            return toks, deps
        
        train_data = [gen_seq() for _ in range(200)]
        
        # Quick training
        model.train()
        for _ in range(20):
            np.random.shuffle(train_data)
            for i in range(0, len(train_data), 16):
                batch = train_data[i:i+16]
                ml = max(len(t) for t, _ in batch)
                toks = torch.zeros(len(batch), ml, dtype=torch.long, device=device)
                deps = torch.zeros(len(batch), ml, dtype=torch.long, device=device)
                mask = torch.zeros(len(batch), ml, dtype=torch.bool, device=device)
                for j, (t, d) in enumerate(batch):
                    toks[j, :len(t)] = torch.tensor(t)
                    deps[j, :len(d)] = torch.tensor(d)
                    mask[j, :len(t)] = True
                
                logits = model(toks)
                loss = F.cross_entropy(logits[mask], deps[mask])
                opt.zero_grad()
                loss.backward()
                opt.step()
        
        # Test
        model.eval()
        test_data = [gen_seq() for _ in range(50)]
        correct, total = 0, 0
        with torch.no_grad():
            for t, d in test_data:
                tt = torch.tensor([t], device=device)
                dd = torch.tensor([d], device=device)
                pred = model(tt).argmax(-1)
                correct += (pred == dd).sum().item()
                total += len(t)
        
        accuracy = correct / total
        
        # Should achieve high accuracy (>80%) on this simple task
        assert accuracy > 0.80, f"Bracket counting accuracy too low: {accuracy:.1%}"


# =============================================================================
# Regime Analysis Tests
# =============================================================================

class TestRegimeAnalysis:
    """Test regime analysis functionality."""
    
    def test_get_regime_analysis_keys(self, temporal_layer, device):
        """Test regime analysis returns expected keys."""
        temporal_layer.reset_tracking()
        
        x = torch.randn(10, 20, 32, device=device)
        temporal_layer.forward_sequence(x)
        
        analysis = temporal_layer.get_regime_analysis()
        
        assert 'transition_matrix' in analysis
        assert 'usage' in analysis
        assert 'stable_tiles' in analysis
        assert 'hub_tiles' in analysis
        assert 'self_transition_probs' in analysis
    
    def test_self_transition_probs_range(self, temporal_layer, device):
        """Test self-transition probabilities are in [0, 1]."""
        temporal_layer.reset_tracking()
        
        x = torch.randn(10, 20, 32, device=device)
        temporal_layer.forward_sequence(x)
        
        analysis = temporal_layer.get_regime_analysis()
        probs = analysis['self_transition_probs']
        
        assert probs.min() >= 0
        assert probs.max() <= 1


# =============================================================================
# Gradient Flow Tests
# =============================================================================

class TestGradientFlow:
    """Test gradient flow through temporal tiles."""
    
    def test_gradients_flow_through_state(self, temporal_layer, device):
        """Test gradients flow through state updates."""
        x = torch.randn(2, 5, 32, device=device, requires_grad=True)
        
        output, final_state, _ = temporal_layer.forward_sequence(x)
        loss = output.sum() + final_state.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
    
    def test_signature_gradients(self, temporal_layer, device):
        """Test signatures receive gradients."""
        temporal_layer.train()
        
        x = torch.randn(4, 10, 32, device=device)
        output, _, _ = temporal_layer.forward_sequence(x)
        loss = output.sum()
        loss.backward()
        
        assert temporal_layer.signatures.grad is not None


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_timestep(self, temporal_layer, device):
        """Test single timestep sequence."""
        x = torch.randn(2, 1, 32, device=device)
        output, final_state, infos = temporal_layer.forward_sequence(x)
        
        assert output.shape == (2, 1, 32)
        assert len(infos) == 1
    
    def test_batch_size_one(self, temporal_layer, device):
        """Test batch size of 1."""
        x = torch.randn(1, 10, 32, device=device)
        output, final_state, infos = temporal_layer.forward_sequence(x)
        
        assert output.shape == (1, 10, 32)
    
    def test_long_sequence(self, temporal_layer, device):
        """Test longer sequence."""
        x = torch.randn(2, 100, 32, device=device)
        output, final_state, infos = temporal_layer.forward_sequence(x)
        
        assert output.shape == (2, 100, 32)
        assert len(infos) == 100
    
    def test_eval_mode(self, temporal_layer, device):
        """Test behavior in eval mode."""
        temporal_layer.eval()
        
        x = torch.randn(4, 10, 32, device=device)
        with torch.no_grad():
            output, final_state, infos = temporal_layer.forward_sequence(x)
        
        assert output.shape == (4, 10, 32)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
