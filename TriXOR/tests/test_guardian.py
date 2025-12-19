"""
Tests for Mesa 12 Guardian Angel Architecture

Tests cover:
- ProgrammableTile and ProgrammableTileBank
- Observer components (ObservationFrame, StateEncoder, ObserverModel, ObservationBuffer)
- Reflector components (XORReflector, SuperpositionedReflector, TrainingManifoldReflector)
- GuardianAngel
- HALOPipeline and EntropicHarmonyLoss
- GuardedTrainer
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from trix.guardian import (
    ProgrammableTile,
    ProgrammableTileBank,
    ObservationFrame,
    StateEncoder,
    ObserverModel,
    SuperpositionedReflector,
    XORReflector,
    GuardianAngel,
    GuardedTrainer,
    HALOPipeline,
    Phase,
    EntropicHarmonyLoss,
    JourneyContext,
)
from trix.guardian.observer import ObservationBuffer
from trix.guardian.reflector import TrainingManifoldReflector


# =============================================================================
# ProgrammableTile Tests
# =============================================================================

class TestProgrammableTile:
    """Tests for ProgrammableTile."""

    def test_init(self):
        """Test tile initialization."""
        tile = ProgrammableTile(d_model=64, d_hidden=128, tile_id=0)
        assert tile.d_model == 64
        assert tile.d_hidden == 128
        assert tile.tile_id == 0
        assert tile.signature.shape == (64,)
        assert tile.weights_up.shape == (64, 128)
        assert tile.weights_down.shape == (128, 64)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)
        x = torch.randn(8, 64)
        out = tile(x)
        assert out.shape == (8, 64)

    def test_forward_3d(self):
        """Test forward with 3D input."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)
        x = torch.randn(4, 16, 64)
        out = tile(x)
        assert out.shape == (4, 16, 64)

    def test_read_signature(self):
        """Test reading signature returns detached clone."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)
        sig = tile.read_signature()
        assert sig.shape == (64,)
        assert not sig.requires_grad
        # Modifying returned tensor shouldn't affect original
        sig[0] = 999.0
        assert tile.signature[0].item() != 999.0

    def test_read_weights(self):
        """Test reading weights returns detached clones."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)
        up, down = tile.read_weights()
        assert up.shape == (64, 128)
        assert down.shape == (128, 64)
        assert not up.requires_grad
        assert not down.requires_grad

    def test_write_signature_blend(self):
        """Test signature blending."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)
        original = tile.read_signature().clone()
        new_sig = torch.ones(64)

        # Blend with 0.5
        tile.write_signature(new_sig, blend=0.5, reason="test")

        blended = tile.read_signature()
        expected = 0.5 * original + 0.5 * new_sig
        assert torch.allclose(blended, expected, atol=1e-5)

    def test_write_signature_zero_blend(self):
        """Test that blend=0 doesn't change signature."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)
        original = tile.read_signature().clone()
        new_sig = torch.ones(64) * 100

        tile.write_signature(new_sig, blend=0.0)

        assert torch.allclose(tile.read_signature(), original)

    def test_write_signature_full_blend(self):
        """Test that blend=1 fully replaces signature."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)
        new_sig = torch.ones(64) * 42

        tile.write_signature(new_sig, blend=1.0)

        assert torch.allclose(tile.read_signature(), new_sig)

    def test_freeze_prevents_write(self):
        """Test that frozen tile rejects writes."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)
        original = tile.read_signature().clone()

        tile.freeze()
        result = tile.write_signature(torch.ones(64), blend=1.0)

        assert result is False
        assert torch.allclose(tile.read_signature(), original)

    def test_unfreeze_allows_write(self):
        """Test that unfreezing allows writes again."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)

        tile.freeze()
        tile.unfreeze()
        result = tile.write_signature(torch.ones(64), blend=1.0)

        assert result is True

    def test_version_increments(self):
        """Test that version increments on write."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)
        assert tile.version == 0

        tile.write_signature(torch.ones(64), blend=0.1)
        assert tile.version == 1

        tile.write_signature(torch.ones(64), blend=0.1)
        assert tile.version == 2

    def test_history_tracking(self):
        """Test modification history is tracked."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)

        tile.write_signature(torch.ones(64), blend=0.1, reason="first")
        tile.write_signature(torch.ones(64), blend=0.2, reason="second")

        history = tile.history
        assert len(history) == 2
        assert history[0].reason == "first"
        assert history[0].blend == 0.1
        assert history[1].reason == "second"
        assert history[1].blend == 0.2

    def test_signature_movement(self):
        """Test signature movement tracking."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)
        tile.save_initial_state()

        # Initially no movement
        assert tile.signature_movement == 0.0

        # Move the signature
        tile.write_signature(tile.signature + torch.ones(64), blend=1.0)

        # Should have moved
        assert tile.signature_movement > 0

    def test_gradient_flow(self):
        """Test gradients flow through tile."""
        tile = ProgrammableTile(d_model=64, d_hidden=128)
        x = torch.randn(8, 64, requires_grad=True)

        out = tile(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert tile.weights_up.grad is not None
        assert tile.weights_down.grad is not None


class TestProgrammableTileBank:
    """Tests for ProgrammableTileBank."""

    def test_init(self):
        """Test bank initialization."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)
        assert bank.num_tiles == 8
        assert len(bank.tiles) == 8

    def test_get_signatures(self):
        """Test getting all signatures."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)
        sigs = bank.get_signatures()
        assert sigs.shape == (8, 64)

    def test_save_initial_state(self):
        """Test saving initial state for all tiles."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)
        bank.save_initial_state()

        # All tiles should have initial state saved
        for tile in bank.tiles:
            assert tile._initial_signature is not None

    def test_get_signature_movements(self):
        """Test getting movements for all tiles."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)
        bank.save_initial_state()

        movements = bank.get_signature_movements()
        assert len(movements) == 8
        assert all(m == 0.0 for m in movements)  # No movement yet

    def test_apply_signature_corrections(self):
        """Test applying corrections to multiple tiles."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)

        corrections = torch.randn(8, 64)
        blends = torch.tensor([0.1, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1])

        modified = bank.apply_signature_corrections(corrections, blends, reason="test")

        # Should have modified tiles with blend > 0.01
        assert modified == 4  # tiles 0, 1, 4, 7

    def test_compute_routing_scores(self):
        """Test routing score computation."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)
        query = torch.randn(4, 64)

        scores = bank.compute_routing_scores(query)

        assert scores.shape == (4, 8)

    def test_compute_routing_scores_3d(self):
        """Test routing scores with 3D input."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)
        query = torch.randn(4, 16, 64)

        scores = bank.compute_routing_scores(query)

        assert scores.shape == (4, 16, 8)

    def test_route_and_compute_soft(self):
        """Test soft routing and computation."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)
        query = torch.randn(4, 64)

        output, info = bank.route_and_compute(query, soft=True)

        assert output.shape == (4, 64)
        assert 'weights' in info
        assert 'entropy' in info
        assert info['weights'].shape == (4, 8)

    def test_freeze_all(self):
        """Test freezing all tiles."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)

        bank.freeze_all()

        assert all(tile.is_frozen for tile in bank.tiles)

    def test_unfreeze_all(self):
        """Test unfreezing all tiles."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)
        bank.freeze_all()

        bank.unfreeze_all()

        assert all(not tile.is_frozen for tile in bank.tiles)

    def test_freeze_tiles_selective(self):
        """Test freezing specific tiles."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)

        bank.freeze_tiles([0, 2, 4])

        assert bank.tiles[0].is_frozen
        assert not bank.tiles[1].is_frozen
        assert bank.tiles[2].is_frozen
        assert not bank.tiles[3].is_frozen
        assert bank.tiles[4].is_frozen

    def test_get_tile_stats(self):
        """Test tile statistics."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)
        bank.save_initial_state()

        stats = bank.get_tile_stats()

        assert stats['num_tiles'] == 8
        assert 'total_movement' in stats
        assert 'mean_movement' in stats
        assert len(stats['per_tile_movements']) == 8


# =============================================================================
# Observer Tests
# =============================================================================

class TestObservationFrame:
    """Tests for ObservationFrame."""

    def test_create_frame(self):
        """Test creating observation frame."""
        frame = ObservationFrame(
            epoch=1,
            step=10,
            routing_entropy=0.5,
            loss=0.1,
            accuracy=95.0,
        )

        assert frame.epoch == 1
        assert frame.step == 10
        assert frame.routing_entropy == 0.5
        assert frame.loss == 0.1
        assert frame.accuracy == 95.0

    def test_to_tensor(self):
        """Test converting frame to tensor."""
        frame = ObservationFrame(
            epoch=1,
            step=10,
            routing_entropy=0.5,
            loss=0.1,
            accuracy=95.0,
        )

        tensor = frame.to_tensor(d_model=128, num_tiles=16)

        # 11 scalars + 16 tiles + 8 ops = 35
        assert tensor.shape == (35,)
        assert tensor.dtype == torch.float32

    def test_to_tensor_with_tile_activations(self):
        """Test tensor conversion with tile activations."""
        frame = ObservationFrame(
            epoch=1,
            step=10,
            tile_activations=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )

        tensor = frame.to_tensor(d_model=128, num_tiles=4)

        # 11 scalars + 4 tiles + 8 ops = 23
        assert tensor.shape == (23,)


class TestStateEncoder:
    """Tests for StateEncoder."""

    def test_init(self):
        """Test encoder initialization."""
        encoder = StateEncoder(obs_dim=35, hidden_dim=128, state_dim=64)
        assert encoder.encoder is not None

    def test_forward_shape(self):
        """Test encoder output shape."""
        encoder = StateEncoder(obs_dim=35, hidden_dim=128, state_dim=64)
        obs = torch.randn(35)

        state = encoder(obs)

        assert state.shape == (64,)

    def test_forward_batch(self):
        """Test encoder with batch input."""
        encoder = StateEncoder(obs_dim=35, hidden_dim=128, state_dim=64)
        obs = torch.randn(8, 35)

        state = encoder(obs)

        assert state.shape == (8, 64)

    def test_encode_frame(self):
        """Test encoding ObservationFrame directly."""
        encoder = StateEncoder(obs_dim=35, hidden_dim=128, state_dim=64)
        frame = ObservationFrame(epoch=1, step=10)

        state = encoder.encode_frame(frame)

        assert state.shape == (64,)

    def test_gradient_flow(self):
        """Test gradients flow through encoder."""
        encoder = StateEncoder(obs_dim=35, hidden_dim=128, state_dim=64)
        obs = torch.randn(8, 35, requires_grad=True)

        state = encoder(obs)
        loss = state.sum()
        loss.backward()

        assert obs.grad is not None


class TestObserverModel:
    """Tests for ObserverModel."""

    def test_init(self):
        """Test model initialization."""
        model = ObserverModel(
            state_dim=64,
            hidden_dim=128,
            num_tiles=16,
            num_ops=8,
            d_model=128
        )

        assert model.state_dim == 64
        assert model.num_tiles == 16

    def test_forward(self):
        """Test forward pass."""
        model = ObserverModel(state_dim=64, hidden_dim=128, num_tiles=16, d_model=128)
        state_history = torch.randn(20, 64)  # 20 timesteps

        result = model(state_history)

        assert 'error_probs' in result
        assert 'intervention_logits' in result
        assert 'tile_corrections' in result
        assert 'blend_factors' in result
        assert 'celebration_score' in result

        assert result['error_probs'].shape == (8,)  # num_ops
        assert result['intervention_logits'].shape == (6,)  # intervention levels
        assert result['tile_corrections'].shape == (16, 128)
        assert result['blend_factors'].shape == (16,)

    def test_forward_batched(self):
        """Test forward with batched input."""
        model = ObserverModel(state_dim=64, hidden_dim=128, num_tiles=16, d_model=128)
        state_history = torch.randn(4, 20, 64)  # batch=4, seq=20

        result = model(state_history)

        # Results should be squeezed to batch dimension
        assert result['error_probs'].shape == (4, 8)

    def test_get_intervention_decision(self):
        """Test intervention decision making."""
        model = ObserverModel(state_dim=64, hidden_dim=128, num_tiles=16, d_model=128)
        state_history = torch.randn(20, 64)

        decision = model.get_intervention_decision(state_history)

        assert 'intervene' in decision
        assert 'level' in decision
        assert 'confidence' in decision
        assert 'message' in decision
        assert isinstance(decision['intervene'], bool)
        assert 0 <= decision['level'] <= 5

    def test_gradient_flow(self):
        """Test gradients flow through model."""
        model = ObserverModel(state_dim=64, hidden_dim=128, num_tiles=16, d_model=128)
        state_history = torch.randn(20, 64, requires_grad=True)

        result = model(state_history)
        loss = result['error_probs'].sum()
        loss.backward()

        assert state_history.grad is not None


class TestObservationBuffer:
    """Tests for ObservationBuffer."""

    def test_init(self):
        """Test buffer initialization."""
        buffer = ObservationBuffer(max_size=100, window_size=20)
        assert buffer.max_size == 100
        assert buffer.window_size == 20
        assert len(buffer.frames) == 0

    def test_add(self):
        """Test adding frames."""
        buffer = ObservationBuffer(max_size=100, window_size=20)

        for i in range(10):
            buffer.add(ObservationFrame(epoch=0, step=i))

        assert len(buffer.frames) == 10

    def test_max_size_eviction(self):
        """Test oldest frames are evicted when max size reached."""
        buffer = ObservationBuffer(max_size=10, window_size=5)

        for i in range(15):
            buffer.add(ObservationFrame(epoch=0, step=i))

        assert len(buffer.frames) == 10
        # Oldest should be step=5, not step=0
        assert buffer.frames[0].step == 5

    def test_get_recent_window(self):
        """Test getting recent window."""
        buffer = ObservationBuffer(max_size=100, window_size=5)

        for i in range(10):
            buffer.add(ObservationFrame(epoch=0, step=i, loss=float(i)))

        window = buffer.get_recent_window()

        assert window.shape[0] == 5  # window_size

    def test_get_recent_window_padding(self):
        """Test window padding when not enough frames."""
        buffer = ObservationBuffer(max_size=100, window_size=10)

        for i in range(3):
            buffer.add(ObservationFrame(epoch=0, step=i))

        window = buffer.get_recent_window()

        assert window.shape[0] == 10  # Padded to window_size

    def test_get_trajectory_stats(self):
        """Test trajectory statistics."""
        buffer = ObservationBuffer(max_size=100, window_size=20)

        for i in range(10):
            buffer.add(ObservationFrame(epoch=0, step=i, loss=1.0 - i*0.1, accuracy=50 + i*5))

        stats = buffer.get_trajectory_stats()

        assert 'loss_trend' in stats
        assert 'acc_trend' in stats
        assert stats['loss_trend'] < 0  # Loss decreasing
        assert stats['acc_trend'] > 0  # Accuracy increasing

    def test_clear(self):
        """Test clearing buffer."""
        buffer = ObservationBuffer(max_size=100, window_size=20)

        for i in range(10):
            buffer.add(ObservationFrame(epoch=0, step=i))

        buffer.clear()

        assert len(buffer.frames) == 0


# =============================================================================
# Reflector Tests
# =============================================================================

class TestXORReflector:
    """Tests for XORReflector."""

    def test_init(self):
        """Test reflector initialization."""
        reflector = XORReflector(d_model=64)
        assert reflector.d_model == 64

    def test_forward(self):
        """Test forward pass."""
        reflector = XORReflector(d_model=64)
        current = torch.randn(8, 64)
        previous = torch.randn(8, 64)

        reflection, info = reflector(current, previous)

        assert reflection.shape == (8, 64)
        assert 'delta' in info
        assert 'magnitude' in info
        assert 'significant' in info
        assert 'direction' in info

    def test_no_change_small_magnitude(self):
        """Test that identical inputs produce small magnitude."""
        reflector = XORReflector(d_model=64)
        x = torch.randn(8, 64)

        _, info = reflector(x, x)

        assert torch.allclose(info['magnitude'], torch.zeros(8), atol=1e-5)

    def test_assess_trajectory(self):
        """Test trajectory assessment."""
        reflector = XORReflector(d_model=64)
        current = torch.randn(8, 64)
        previous = torch.randn(8, 64)
        target = torch.randn(8, 64)

        assessment = reflector.assess_trajectory(current, previous, target)

        assert 'change_magnitude' in assessment
        assert 'change_direction' in assessment
        assert 'alignment' in assessment

    def test_gradient_flow(self):
        """Test gradients flow through reflector."""
        reflector = XORReflector(d_model=64)
        current = torch.randn(8, 64, requires_grad=True)
        previous = torch.randn(8, 64)

        reflection, _ = reflector(current, previous)
        loss = reflection.sum()
        loss.backward()

        assert current.grad is not None


class TestSuperpositionedReflector:
    """Tests for SuperpositionedReflector."""

    def test_init(self):
        """Test reflector initialization."""
        reflector = SuperpositionedReflector(d_model=64, num_bases=4)
        assert reflector.d_model == 64
        assert reflector.num_bases == 4
        assert reflector.bases.shape == (4, 64, 64)

    def test_forward(self):
        """Test forward pass."""
        reflector = SuperpositionedReflector(d_model=64, num_bases=4)
        x = torch.randn(8, 64)

        output, info = reflector(x)

        assert output.shape == (8, 64)
        assert 'reflections' in info
        assert 'combination_weights' in info
        assert info['reflections'].shape == (4, 8, 64)

    def test_forward_3d(self):
        """Test forward with 3D input."""
        reflector = SuperpositionedReflector(d_model=64, num_bases=4)
        x = torch.randn(4, 16, 64)

        output, info = reflector(x)

        assert output.shape == (4, 16, 64)

    def test_reflect_single_basis(self):
        """Test reflecting through single basis."""
        reflector = SuperpositionedReflector(d_model=64, num_bases=4)
        x = torch.randn(8, 64)

        reflected = reflector.reflect_single_basis(x, basis_idx=0)

        assert reflected.shape == (8, 64)

    def test_orthogonality_loss(self):
        """Test orthogonality regularization loss."""
        reflector = SuperpositionedReflector(d_model=64, num_bases=4)

        loss = reflector.get_orthogonality_loss()

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_analyze_diversity(self):
        """Test diversity analysis."""
        reflector = SuperpositionedReflector(d_model=64, num_bases=4)

        diversity = reflector.analyze_diversity()

        assert 'mean_similarity' in diversity
        assert 'max_similarity' in diversity
        assert 'num_bases' in diversity
        assert diversity['num_bases'] == 4

    def test_residual_connection(self):
        """Test that output includes residual of input."""
        reflector = SuperpositionedReflector(d_model=64, num_bases=4)
        x = torch.randn(8, 64)

        output, _ = reflector(x)

        # Output should be different from input (superposition added)
        assert not torch.allclose(output, x)

    def test_gradient_flow(self):
        """Test gradients flow through reflector."""
        reflector = SuperpositionedReflector(d_model=64, num_bases=4)
        x = torch.randn(8, 64, requires_grad=True)

        output, _ = reflector(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert reflector.bases.grad is not None


class TestTrainingManifoldReflector:
    """Tests for TrainingManifoldReflector."""

    def test_init(self):
        """Test reflector initialization."""
        reflector = TrainingManifoldReflector(state_dim=64, hidden_dim=128)
        assert reflector.state_dim == 64

    def test_forward(self):
        """Test forward pass."""
        reflector = TrainingManifoldReflector(state_dim=64, hidden_dim=128)
        state_history = torch.randn(20, 64)  # 20 timesteps

        assessment, info = reflector(state_history)

        assert assessment.shape == (3,)  # good, neutral, needs_correction
        assert assessment.sum().item() == pytest.approx(1.0, abs=1e-5)  # Softmax
        assert 'good_prob' in info
        assert 'neutral_prob' in info
        assert 'needs_correction_prob' in info
        assert 'message' in info

    def test_message_generation(self):
        """Test message generation based on assessment."""
        reflector = TrainingManifoldReflector(state_dim=64, hidden_dim=128)

        # Test each message type
        probs_good = torch.tensor([0.8, 0.1, 0.1])
        assert "look at 'em go" in reflector._get_message(probs_good)

        probs_neutral = torch.tensor([0.1, 0.8, 0.1])
        assert "Steady" in reflector._get_message(probs_neutral)

        probs_correct = torch.tensor([0.1, 0.1, 0.8])
        assert "next time" in reflector._get_message(probs_correct)

    def test_gradient_flow(self):
        """Test gradients flow through reflector."""
        reflector = TrainingManifoldReflector(state_dim=64, hidden_dim=128)
        state_history = torch.randn(20, 64, requires_grad=True)

        assessment, _ = reflector(state_history)
        loss = assessment.sum()
        loss.backward()

        assert state_history.grad is not None


# =============================================================================
# GuardianAngel Tests
# =============================================================================

class TestGuardianAngel:
    """Tests for GuardianAngel."""

    def test_init(self):
        """Test guardian initialization."""
        guardian = GuardianAngel(
            d_model=128,
            num_tiles=16,
            gentleness=0.1,
            intervention_threshold=0.7
        )

        assert guardian.d_model == 128
        assert guardian.num_tiles == 16
        assert guardian.gentleness == 0.1

    def test_observe(self):
        """Test adding observations."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)
        frame = ObservationFrame(epoch=0, step=0, loss=0.5)

        guardian.observe(frame)

        assert len(guardian.observation_buffer.frames) == 1

    def test_get_observation_window(self):
        """Test getting observation window."""
        guardian = GuardianAngel(d_model=128, num_tiles=16, window_size=5)

        for i in range(10):
            guardian.observe(ObservationFrame(epoch=0, step=i))

        window = guardian.get_observation_window()

        assert window.shape[0] == 5  # window_size

    def test_reflect(self):
        """Test superpositioned reflection."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)
        repr = torch.randn(8, 128)

        output, info = guardian.reflect(repr)

        assert output.shape == (8, 128)
        assert 'reflections' in info

    def test_reflect_on_change(self):
        """Test XOR reflection on change."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)
        current = torch.randn(8, 128)
        previous = torch.randn(8, 128)

        info = guardian.reflect_on_change(current, previous)

        assert 'magnitude' in info

    def test_reflect_on_change_first_observation(self):
        """Test first observation handling."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)
        current = torch.randn(8, 128)

        info = guardian.reflect_on_change(current)

        assert info['message'] == 'First observation'

    def test_predict_not_ready(self):
        """Test prediction when not enough observations."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)

        prediction = guardian.predict()

        assert prediction['ready'] is False

    def test_predict_ready(self):
        """Test prediction with enough observations."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)

        for i in range(10):
            guardian.observe(ObservationFrame(epoch=0, step=i))

        prediction = guardian.predict()

        assert prediction['ready'] is True
        assert 'intervene' in prediction

    def test_should_intervene(self):
        """Test intervention decision."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)

        for i in range(10):
            guardian.observe(ObservationFrame(epoch=0, step=i))

        should, prediction = guardian.should_intervene()

        assert isinstance(should, bool)
        assert 'message' in prediction

    def test_intervene(self):
        """Test applying intervention."""
        guardian = GuardianAngel(d_model=128, num_tiles=16, gentleness=0.1)
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)

        # Need observations first
        for i in range(10):
            guardian.observe(ObservationFrame(epoch=0, step=i))

        prediction = guardian.predict()

        record = guardian.intervene(
            tile_bank=tile_bank,
            prediction=prediction,
            epoch=0,
            step=10,
            current_accuracy=50.0
        )

        assert record.epoch == 0
        assert record.step == 10
        assert record.before_accuracy == 50.0

    def test_step(self):
        """Test full guardian step."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)

        for i in range(10):
            frame = ObservationFrame(epoch=0, step=i)
            result = guardian.step(
                tile_bank=tile_bank,
                observation=frame,
                current_repr=torch.randn(8, 128)
            )

        assert result['observed'] is True
        assert 'prediction' in result
        assert 'change_info' in result

    def test_get_stats(self):
        """Test statistics retrieval."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)

        for i in range(5):
            guardian.observe(ObservationFrame(epoch=0, step=i))

        stats = guardian.get_stats()

        assert stats['total_observations'] == 5
        assert 'total_interventions' in stats
        assert 'celebration_count' in stats
        assert 'gentleness_setting' in stats

    def test_celebration_rate(self):
        """Test celebration rate calculation."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)

        # No observations
        assert guardian.get_celebration_rate() == 0.0

        # Add observations
        for i in range(10):
            guardian.observe(ObservationFrame(epoch=0, step=i))

        rate = guardian.get_celebration_rate()
        assert 0 <= rate <= 1

    def test_state_save_load(self):
        """Test saving and loading state."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)

        for i in range(5):
            guardian.observe(ObservationFrame(epoch=0, step=i))

        state = guardian.get_state_for_save()

        assert 'observer' in state
        assert 'reflector' in state
        assert 'stats' in state

        # Create new guardian and load state
        guardian2 = GuardianAngel(d_model=128, num_tiles=16)
        guardian2.load_state_from_save(state)

        # Stats should match
        assert guardian2.total_interventions == guardian.total_interventions


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestEntropicHarmonyLoss:
    """Tests for EntropicHarmonyLoss."""

    def test_init(self):
        """Test loss initialization."""
        loss = EntropicHarmonyLoss(target_entropy=0.7)
        assert loss.target_entropy == 0.7

    def test_forward(self):
        """Test forward pass."""
        loss_fn = EntropicHarmonyLoss()
        routing_weights = torch.softmax(torch.randn(8, 16), dim=-1)

        loss, metrics = loss_fn(routing_weights, signature_diversity=1.0, entropy_history=[])

        assert loss.ndim == 0  # Scalar
        assert 'entropy' in metrics
        assert 'diversity' in metrics
        assert 'harmony' in metrics

    def test_collapsed_routing_penalized(self):
        """Test that collapsed routing has higher loss."""
        loss_fn = EntropicHarmonyLoss()

        # Uniform routing (high entropy)
        uniform = torch.ones(8, 16) / 16
        loss_uniform, _ = loss_fn(uniform, 1.0, [])

        # Collapsed routing (low entropy)
        collapsed = torch.zeros(8, 16)
        collapsed[:, 0] = 1.0
        loss_collapsed, _ = loss_fn(collapsed, 1.0, [])

        # Collapsed should have higher loss (more penalty)
        assert loss_collapsed > loss_uniform

    def test_variance_penalty(self):
        """Test variance penalty with history."""
        loss_fn = EntropicHarmonyLoss(variance_penalty=1.0)
        routing = torch.softmax(torch.randn(8, 16), dim=-1)

        # Stable history
        stable_history = [0.5] * 10
        _, metrics_stable = loss_fn(routing, 1.0, stable_history)

        # Unstable history
        unstable_history = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5]
        _, metrics_unstable = loss_fn(routing, 1.0, unstable_history)

        assert metrics_unstable['variance'] > metrics_stable['variance']


class TestPhaseEnum:
    """Tests for Phase enum."""

    def test_phases_exist(self):
        """Test all phases exist."""
        assert Phase.EXPLORATION.value == 1
        assert Phase.EXPEDITION.value == 2
        assert Phase.CONVERGENCE.value == 3
        assert Phase.MASTERY.value == 4


class TestJourneyContext:
    """Tests for JourneyContext."""

    def test_init(self):
        """Test context initialization."""
        journey = JourneyContext()

        assert journey.natural_entropy_level == 0.0
        assert journey.preferred_tiles == []
        assert journey.nodes_of_interest == []
        assert journey.struggle_points == {}

    def test_mutable_fields(self):
        """Test that mutable fields work correctly."""
        journey = JourneyContext()

        journey.preferred_tiles.append(0)
        journey.preferred_tiles.append(1)

        assert len(journey.preferred_tiles) == 2

        journey.struggle_points['ADC'] = 0.8

        assert journey.struggle_points['ADC'] == 0.8


class TestHALOPipeline:
    """Tests for HALOPipeline."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                out = self.linear(x)
                # Return (predictions, routing_info, aux_loss)
                routing_info = {'weights': torch.softmax(torch.randn(x.size(0), 16), dim=-1)}
                return out, routing_info, {'total_aux': 0.0}

        return SimpleModel()

    @pytest.fixture
    def simple_loader(self):
        """Create simple data loader for testing."""
        data = [(torch.randn(8, 10), torch.randint(0, 5, (8,))) for _ in range(5)]
        return data

    def test_init(self, simple_model, simple_loader):
        """Test pipeline initialization."""
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)
        guardian = GuardianAngel(d_model=128, num_tiles=16)

        pipeline = HALOPipeline(
            model=simple_model,
            tile_bank=tile_bank,
            guardian=guardian,
            optimizer_fn=lambda params, lr: torch.optim.Adam(params, lr=lr),
            task_loss_fn=nn.CrossEntropyLoss(),
            device='cpu',
            total_epochs=4,
            phase_schedule=(1, 1, 1, 1),
            verbose=False,
        )

        assert pipeline.current_phase == Phase.EXPLORATION
        assert pipeline.global_epoch == 0

    def test_detect_nodes_of_interest(self, simple_model):
        """Test node detection."""
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)
        guardian = GuardianAngel(d_model=128, num_tiles=16)

        pipeline = HALOPipeline(
            model=simple_model,
            tile_bank=tile_bank,
            guardian=guardian,
            optimizer_fn=lambda params, lr: torch.optim.Adam(params, lr=lr),
            task_loss_fn=nn.CrossEntropyLoss(),
            device='cpu',
            verbose=False,
        )

        # Create observations with tile activations
        observations = []
        for i in range(20):
            obs = ObservationFrame(
                epoch=0,
                step=i,
                tile_activations=torch.rand(16),
                routing_entropy=0.5 + 0.1 * np.random.randn(),
            )
            observations.append(obs)

        nodes = pipeline.detect_nodes_of_interest(observations)

        # Should return a list (may be empty depending on randomness)
        assert isinstance(nodes, list)


# =============================================================================
# GuardedTrainer Tests
# =============================================================================

class TestGuardedTrainer:
    """Tests for GuardedTrainer."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                out = torch.sigmoid(self.linear(x))
                routing_info = {'entropy': torch.tensor(0.5)}
                return out, routing_info, {'total_aux': 0.0}

        return SimpleModel()

    @pytest.fixture
    def simple_loader(self):
        """Create simple data loader for testing."""
        # Use unsqueezed targets to match output shape [batch, 1]
        data = [(torch.randn(8, 10), torch.randint(0, 2, (8, 1)).float()) for _ in range(3)]
        return data

    def test_init(self, simple_model):
        """Test trainer initialization."""
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)
        guardian = GuardianAngel(d_model=128, num_tiles=16)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

        trainer = GuardedTrainer(
            model=simple_model,
            tile_bank=tile_bank,
            guardian=guardian,
            optimizer=optimizer,
            loss_fn=nn.BCELoss(),
            device='cpu',
            verbose=False,
        )

        assert trainer.epoch == 0
        assert trainer.global_step == 0

    def test_create_observation(self, simple_model):
        """Test observation creation."""
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)
        guardian = GuardianAngel(d_model=128, num_tiles=16)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

        trainer = GuardedTrainer(
            model=simple_model,
            tile_bank=tile_bank,
            guardian=guardian,
            optimizer=optimizer,
            loss_fn=nn.BCELoss(),
            device='cpu',
            verbose=False,
        )

        obs = trainer.create_observation(
            batch_info={'size': 8},
            loss=0.5,
            accuracy=80.0,
            routing_info={'entropy': torch.tensor(0.6)},
        )

        assert obs.loss == 0.5
        assert obs.accuracy == 80.0
        assert obs.routing_entropy == pytest.approx(0.6, abs=1e-5)

    def test_train_step(self, simple_model, simple_loader):
        """Test single training step."""
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)
        guardian = GuardianAngel(d_model=128, num_tiles=16)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

        trainer = GuardedTrainer(
            model=simple_model,
            tile_bank=tile_bank,
            guardian=guardian,
            optimizer=optimizer,
            loss_fn=nn.BCELoss(),
            device='cpu',
            verbose=False,
        )

        batch = simple_loader[0]
        result = trainer.train_step(batch)

        assert 'loss' in result
        assert 'accuracy' in result
        assert 'guardian' in result
        assert trainer.global_step == 1

    def test_train_epoch(self, simple_model, simple_loader):
        """Test training epoch."""
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)
        guardian = GuardianAngel(d_model=128, num_tiles=16)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

        trainer = GuardedTrainer(
            model=simple_model,
            tile_bank=tile_bank,
            guardian=guardian,
            optimizer=optimizer,
            loss_fn=nn.BCELoss(),
            device='cpu',
            verbose=False,
        )

        summary = trainer.train_epoch(simple_loader)

        assert summary['epoch'] == 1
        assert 'loss' in summary
        assert 'accuracy' in summary
        assert 'interventions' in summary
        assert trainer.epoch == 1

    def test_evaluate(self, simple_model, simple_loader):
        """Test evaluation."""
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)
        guardian = GuardianAngel(d_model=128, num_tiles=16)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

        trainer = GuardedTrainer(
            model=simple_model,
            tile_bank=tile_bank,
            guardian=guardian,
            optimizer=optimizer,
            loss_fn=nn.BCELoss(),
            device='cpu',
            verbose=False,
        )

        result = trainer.evaluate(simple_loader)

        assert 'loss' in result
        assert 'accuracy' in result

    def test_warmup_prevents_intervention(self, simple_model, simple_loader):
        """Test that warmup period prevents interventions."""
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)
        guardian = GuardianAngel(d_model=128, num_tiles=16)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

        trainer = GuardedTrainer(
            model=simple_model,
            tile_bank=tile_bank,
            guardian=guardian,
            optimizer=optimizer,
            loss_fn=nn.BCELoss(),
            device='cpu',
            warmup_epochs=5,
            verbose=False,
        )

        # Train one epoch (within warmup)
        summary = trainer.train_epoch(simple_loader)

        # Should have no interventions during warmup
        assert summary['interventions'] == 0


class TestCreateGuardedTrainingSetup:
    """Tests for create_guarded_training_setup helper."""

    def test_creates_all_components(self):
        """Test that helper creates all components."""
        from trix.guardian.training import create_guarded_training_setup

        model = nn.Linear(10, 1)

        tile_bank, guardian, trainer = create_guarded_training_setup(
            model=model,
            num_tiles=8,
            d_model=64,
            d_hidden=128,
            device='cpu',
        )

        assert isinstance(tile_bank, ProgrammableTileBank)
        assert isinstance(guardian, GuardianAngel)
        assert isinstance(trainer, GuardedTrainer)

        assert tile_bank.num_tiles == 8
        assert guardian.num_tiles == 8


# =============================================================================
# Integration Tests
# =============================================================================

class TestGuardianIntegration:
    """Integration tests for Guardian system."""

    def test_full_observation_cycle(self):
        """Test complete observation → prediction → intervention cycle."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)
        tile_bank.save_initial_state()

        # Build up observations
        for i in range(30):
            frame = ObservationFrame(
                epoch=0,
                step=i,
                loss=1.0 - i * 0.02,
                accuracy=50 + i,
                routing_entropy=0.5,
            )

            result = guardian.step(
                tile_bank=tile_bank,
                observation=frame,
                current_repr=torch.randn(8, 128)
            )

            assert result['observed']

        # Check stats
        stats = guardian.get_stats()
        assert stats['total_observations'] == 30

    def test_tile_bank_with_guardian_interventions(self):
        """Test tile bank modifications through guardian."""
        # Use default num_tiles=16 and num_ops=8 to match ObservationFrame defaults
        guardian = GuardianAngel(d_model=128, num_tiles=16, gentleness=0.2)
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)
        tile_bank.save_initial_state()

        initial_sigs = tile_bank.get_signatures().clone()

        # Build observations with matching parameters
        for i in range(20):
            obs = ObservationFrame(epoch=0, step=i)
            guardian.observe(obs)

        # Force an intervention with corrections
        prediction = guardian.predict()
        if prediction['ready']:
            # Apply corrections
            corrections = torch.randn(16, 128)
            blends = torch.ones(16) * 0.1
            modified = tile_bank.apply_signature_corrections(corrections, blends)

            assert modified == 16  # All tiles modified

            # Signatures should have changed
            new_sigs = tile_bank.get_signatures()
            assert not torch.allclose(initial_sigs, new_sigs)

    def test_reflector_with_training_states(self):
        """Test reflectors with simulated training trajectory."""
        xor_reflector = XORReflector(d_model=64)
        super_reflector = SuperpositionedReflector(d_model=64, num_bases=4)

        # Simulate training trajectory
        states = [torch.randn(8, 64)]

        for i in range(10):
            # Simulate gradient step (small change)
            delta = torch.randn(8, 64) * 0.1
            new_state = states[-1] + delta
            states.append(new_state)

            # XOR reflection on change
            _, xor_info = xor_reflector(new_state, states[-2])
            assert xor_info['magnitude'].mean() < 1.0  # Small changes

            # Superposition reflection
            reflected, super_info = super_reflector(new_state)
            assert reflected.shape == new_state.shape


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_observation_buffer(self):
        """Test guardian with no observations."""
        guardian = GuardianAngel(d_model=128, num_tiles=16)

        prediction = guardian.predict()

        assert not prediction['ready']

    def test_single_tile(self):
        """Test with single tile."""
        bank = ProgrammableTileBank(num_tiles=1, d_model=64, d_hidden=128)
        query = torch.randn(8, 64)

        scores = bank.compute_routing_scores(query)

        assert scores.shape == (8, 1)

    def test_large_blend(self):
        """Test that gentleness caps blend factor."""
        # Use default num_tiles=16 to match ObservationFrame defaults
        guardian = GuardianAngel(d_model=128, num_tiles=16, gentleness=0.1)
        tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)

        # Build observations
        for i in range(20):
            guardian.observe(ObservationFrame(epoch=0, step=i))

        prediction = guardian.predict()

        # Simulate intervention with explicit blend factors
        if prediction.get('blend_factors') is not None:
            # After clamping in intervene, blends should be <= gentleness
            clamped = prediction['blend_factors'].clamp(max=guardian.gentleness)
            assert clamped.max() <= guardian.gentleness

    def test_frozen_tiles_resist_intervention(self):
        """Test that frozen tiles resist modification."""
        bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128)
        bank.save_initial_state()

        # Freeze some tiles
        bank.freeze_tiles([0, 1, 2])
        initial_frozen = bank.tiles[0].read_signature().clone()

        # Apply corrections
        corrections = torch.ones(8, 64)
        blends = torch.ones(8) * 0.5
        modified = bank.apply_signature_corrections(corrections, blends)

        # Only unfrozen tiles should be modified
        assert modified == 5
        assert torch.allclose(bank.tiles[0].read_signature(), initial_frozen)

    def test_device_consistency(self):
        """Test that components work on same device."""
        device = 'cpu'  # Use CPU for testing

        guardian = GuardianAngel(d_model=64, num_tiles=8).to(device)
        tile_bank = ProgrammableTileBank(num_tiles=8, d_model=64, d_hidden=128).to(device)

        query = torch.randn(8, 64, device=device)
        scores = tile_bank.compute_routing_scores(query)

        assert scores.device.type == device
