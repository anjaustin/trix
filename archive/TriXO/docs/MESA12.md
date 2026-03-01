# Mesa 12: Training Observer Architecture

**Status: Experimental / Research**

This module provides infrastructure for observing and adapting training dynamics.
The core observation layer is implemented and tested; the adaptive learning loop
is incomplete and requires validation.

---

## What Mesa 12 Is

A system for monitoring neural network training with the capability to intervene:

1. **Observation**: Track routing patterns, signature movement, gradient flow, loss trajectories
2. **Reflection**: Analyze what changed between states (XOR delta), project through multiple bases
3. **Prediction**: Anticipate where training will struggle
4. **Intervention**: Apply bounded corrections to tile signatures

## What's Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| `ProgrammableTile` | Working | Read/write interface for signatures and weights |
| `ProgrammableTileBank` | Working | Bulk operations, routing, statistics |
| `ObservationFrame` | Working | Captures training dynamics per step |
| `ObservationBuffer` | Working | Windowed history management |
| `StateEncoder` | Working | Encodes observations to fixed-size vectors |
| `ObserverModel` | Working | LSTM-based prediction (untrained) |
| `XORReflector` | Working | Delta analysis between states |
| `SuperpositionedReflector` | Working | Multi-basis projection |
| `TrainingManifoldReflector` | Working | Trajectory assessment (untrained) |
| `TrainingObserver` | Working | Integration of above components |
| `AdaptiveTrainingPipeline` | Working | 4-phase training structure |
| `GuardedTrainer` | Working | Training loop with observation |

All components pass unit tests (101 tests total).

## What's Not Yet Validated

### The Learning Loop

The observer can predict and intervene, but:

- **No training signal**: The observer models are randomly initialized. There's no
  feedback loop where intervention outcomes update the observer.
- **No A/B validation**: No rigorous comparison of training with vs. without observer
  to prove intervention helps.
- **Untrained predictors**: `ObserverModel` and `TrainingManifoldReflector` make
  predictions, but haven't been trained on actual training trajectories.

### Open Questions

1. **Does intervention help?** This is the fundamental question. The infrastructure
   exists; the proof does not.

2. **How to train the observer?** Chicken-and-egg: need intervention outcomes to
   train, need trained observer to get good outcomes.

3. **What's the minimal effective observer?** Current system has many components.
   Which are necessary?

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TrainingObserver                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
│  │   Observe    │ → │   Reflect    │ → │   Predict    │     │
│  │              │   │              │   │              │     │
│  │ ObservationFrame  XORReflector    ObserverModel    │     │
│  │ StateEncoder │   │ Superpos.    │   │              │     │
│  └──────────────┘   └──────────────┘   └──────────────┘     │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                     Intervene                        │    │
│  │                                                      │    │
│  │  Level 0: None                                       │    │
│  │  Level 1-3: Gradient/LR adjustments (not impl.)     │    │
│  │  Level 4-5: Signature/weight surgery                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               ProgrammableTileBank                   │    │
│  │                                                      │    │
│  │   [T0] [T1] [T2] ... [Tn]                           │    │
│  │                                                      │    │
│  │   Blend-based writes (not replacement)              │    │
│  │   Version tracking, freeze/unfreeze                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Usage

### Basic Observation

```python
from trix.guardian import (
    TrainingObserver,
    ProgrammableTileBank,
    ObservationFrame,
)

# Create components
tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)
observer = TrainingObserver(d_model=128, num_tiles=16)

# During training, create observation frames
frame = ObservationFrame(
    epoch=0,
    step=100,
    loss=0.5,
    accuracy=75.0,
    routing_entropy=2.3,
)

# Observer step: observe, reflect, potentially intervene
result = observer.step(
    tile_bank=tile_bank,
    observation=frame,
    current_repr=hidden_states,  # Optional
)

print(result['message'])  # Observer status
print(result['intervened'])  # Whether intervention occurred
```

### With GuardedTrainer

```python
from trix.guardian import create_guarded_training_setup

tile_bank, observer, trainer = create_guarded_training_setup(
    model=your_model,
    num_tiles=16,
    d_model=128,
    max_blend=0.1,  # Cap on intervention strength
)

# Train with observation
results = trainer.train(
    train_loader=train_loader,
    eval_loader=eval_loader,
    epochs=30,
)

print(results['observer_stats'])
```

### Adaptive Pipeline

```python
from trix.guardian import AdaptiveTrainingPipeline, Phase

pipeline = AdaptiveTrainingPipeline(
    model=model,
    tile_bank=tile_bank,
    observer=observer,
    optimizer=optimizer,
)

# 4-phase training: Exploration → Expedition → Convergence → Mastery
for batch in train_loader:
    loss, phase_info = pipeline.step(batch, targets)

    if pipeline.phase == Phase.MASTERY:
        print("Reached mastery phase")
```

---

## Key Design Decisions

### Bounded Intervention

All interventions use blend factors (0-1) to interpolate between current and target values:

```python
new_value = (1 - blend) * old_value + blend * correction
```

Default `max_blend=0.1` means at most 10% change per intervention. This prevents
destabilizing learned representations.

### Observation Transparency

TriX's ternary structure makes training dynamics observable:
- Routing scores show which tiles are selected
- Signatures are inspectable vectors
- Tile activations are countable

This transparency is what makes the observer architecture possible.

### Minimal Intervention Principle

The observer should be lazy. Every intervention has a cost. The system is designed
to intervene only when:
- Confidence is high (default threshold: 0.7)
- Training isn't already succeeding
- Warmup period has passed

---

## What Would Validate This

1. **A/B Comparison**: Same model, same task, same seed - train with and without
   observer. Measure final accuracy and training stability.

2. **Learning Signal**: Implement feedback loop where intervention outcomes
   (accuracy delta after intervention) update observer weights.

3. **Ablation Study**: Which components contribute? Can we get benefit with
   simpler observation?

4. **Generalization Test**: Train observer on some seeds, test on held-out seeds.
   Does it generalize?

---

## Files

```
src/trix/guardian/
├── __init__.py              # Public API
├── programmable_tile.py     # ProgrammableTile, ProgrammableTileBank
├── observer.py              # ObservationFrame, StateEncoder, ObserverModel
├── reflector.py             # XORReflector, SuperpositionedReflector
├── guardian.py              # TrainingObserver (main integration)
├── pipeline.py              # AdaptiveTrainingPipeline, phases
└── training.py              # GuardedTrainer

tests/
└── test_guardian.py         # 101 tests covering all components
```

---

## Backwards Compatibility

Old names are aliased for compatibility:

| Old Name | New Name |
|----------|----------|
| `GuardianAngel` | `TrainingObserver` |
| `HALOPipeline` | `AdaptiveTrainingPipeline` |
| `EntropicHarmonyLoss` | `EntropyBalanceLoss` |

---

## Relationship to Core TriX

The `guardian/` module is **experimental and optional**. The core TriX layers
(`trix.nn`, `trix.kernel`, `trix.qat`) work independently and are production-ready.

Use the guardian module for:
- Research into adaptive training
- Understanding training dynamics
- Experimenting with meta-learning approaches

Do not use for production without validation.

---

## Citation

If you use this module in research, please note its experimental status.

---

*December 2024*
