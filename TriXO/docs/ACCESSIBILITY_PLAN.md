# Accessibility Plan: Making TriX Usable For Everyone

This document outlines concrete steps to make the TriX project
accessible to universities, hobbyists, and practitioners without
losing technical substance.

---

## Guiding Principles

1. **Clarity over cleverness.** Say what things do in plain terms.
2. **Layers of depth.** Simple first, complex available.
3. **Honest framing.** Distinguish proven from experimental.
4. **Examples that work.** Runnable code beats explanation.

---

## Phase 1: Language Cleanup

### Files Requiring Docstring/Comment Revision

**guardian/__init__.py**
- Remove: "Who needs Human Reinforcement Learning Feedback..."
- Remove: "RLHF is dead. Long live HALO."
- Keep: Technical description of what the module provides

**guardian/guardian.py**
- Rename class: `GuardianAngel` → `TrainingObserver`
- Remove: All "Love" and "gentleness fortified" language
- Replace: Spiritual metaphors with technical descriptions
- Keep: The actual observation/intervention logic

**guardian/observer.py**
- Keep: Technical content is already reasonable
- Remove: Philosophical quotes in docstrings

**guardian/pipeline.py**
- Rename: `HALOPipeline` → `AdaptiveTrainingPipeline`
- Rename: `EntropicHarmonyLoss` → `EntropyBalanceLoss`
- Remove: "RLHF is dead" messaging
- Remove: Emoji from code ("🔥", etc.)
- Keep: The 4-phase structure (rename phases to descriptive terms)

**guardian/programmable_tile.py**
- Content is already technical and clean
- Minor: "gentleness" parameter could become "blend_cap" or similar

**guardian/reflector.py**
- Content is mostly technical
- Remove: Poetic docstring openings
- Keep: Technical explanations of XOR and superposition

**guardian/training.py**
- Remove: "Love as the process" framing
- Keep: Training loop integration logic

### Naming Conventions

| Current | Proposed |
|---------|----------|
| GuardianAngel | TrainingObserver |
| HALOPipeline | AdaptiveTrainingPipeline |
| EntropicHarmonyLoss | EntropyBalanceLoss |
| gentleness | max_blend / blend_cap |
| celebration_count | success_detections |
| Mesa 12 | (remove numbering, describe features) |

---

## Phase 2: Documentation Restructure

### New Documentation Structure

```
docs/
├── README.md              # Project overview (rewrite)
├── QUICKSTART.md          # 5-minute getting started
├── TUTORIAL.md            # Learning-oriented walkthrough
├── ARCHITECTURE.md        # How it works (keep, revise)
├── API.md                 # Reference (keep, revise)
├── BENCHMARKS.md          # Performance data (expand)
├── THEORY.md              # Formal claims (keep, mark rigorous)
├── EXPERIMENTAL.md        # Research directions (new)
└── CHANGELOG.md           # Version history
```

### README.md Structure

```markdown
# TriX: Ternary Routing for Efficient Neural Networks

## What TriX Does
- 16x memory compression via 2-bit weights
- Sparse computation with learned-free routing
- Drop-in replacements for standard layers

## Quick Example
[10 lines of working code]

## Installation
[pip install]

## When To Use TriX
- Deploying models on edge devices
- Reducing memory footprint
- Experimenting with sparse architectures

## When Not To Use TriX
- Maximum accuracy is critical
- Standard hardware with no memory pressure

## Documentation
- [Quickstart](QUICKSTART.md) - Get running in 5 minutes
- [Tutorial](TUTORIAL.md) - Understand how it works
- [API Reference](API.md) - Full documentation

## Experimental Features
The `trix.observer` module contains research code for adaptive
training. See [EXPERIMENTAL.md](EXPERIMENTAL.md) for details.
This is not production-ready.

## Citation
[If published, add citation]
```

### QUICKSTART.md Structure

```markdown
# Quickstart

## Install
pip install trix

## Replace a Linear Layer
[Code example]

## Replace an FFN
[Code example]

## Train with Quantization Awareness
[Code example]

## Export for Deployment
[Code example]

## Next Steps
- [Full tutorial](TUTORIAL.md)
- [API reference](API.md)
```

---

## Phase 3: Code Organization

### Module Renaming

```
src/trix/
├── __init__.py            # Clean public API
├── nn/                    # Neural network layers (keep)
├── kernel/                # Low-level ops (keep)
├── qat/                   # Quantization-aware training (keep)
├── observer/              # Renamed from guardian/
│   ├── __init__.py
│   ├── observer.py        # TrainingObserver (was GuardianAngel)
│   ├── pipeline.py        # AdaptiveTrainingPipeline
│   ├── ...
└── compiler/              # Neural compilation (keep)
```

### Public API (__init__.py)

```python
# Stable API - ready for production
from .nn import TriXLinear, TriXFFN, TriXBlock
from .kernel import pack_weights, unpack_weights
from .qat import TernaryQuantizer, QATTrainer

# Experimental API - research use only
from . import observer  # Clearly separate namespace
```

---

## Phase 4: Examples and Tutorials

### Example Scripts Needed

1. **basic_usage.py** - Simplest possible example
2. **mnist_compression.py** - Compare TriX vs dense on MNIST
3. **language_model.py** - Character-level LM with TriX FFN
4. **edge_deployment.py** - Export and run on constrained device
5. **custom_routing.py** - Advanced: custom tile configurations

### Tutorial Structure

```markdown
# Tutorial: Understanding TriX

## Part 1: Why Ternary?
- The compression argument
- The routing argument
- Tradeoffs

## Part 2: Your First TriX Layer
[Hands-on code]

## Part 3: How Routing Works
- Signatures explained
- Content-addressable lookup
- Why no learned router

## Part 4: Training Strategies
- Straight-through estimator
- Progressive quantization
- When to quantize

## Part 5: Scaling Up
- Hierarchical routing
- Large tile counts
- Performance considerations

## Part 6: Advanced Topics
- Adaptive training (experimental)
- Custom architectures
- Compilation
```

---

## Phase 5: Benchmarks and Evidence

### Required Benchmarks

1. **Memory**: TriX vs dense at various model sizes
2. **Speed**: Inference latency on target hardware
3. **Accuracy**: Perplexity/accuracy vs parameter count
4. **Comparison**: TriX vs MoE vs pruning vs standard quantization

### Benchmark Presentation

Clear tables. Reproducible conditions. Honest about limitations.

```markdown
## MNIST Classification

| Method | Accuracy | Parameters | Memory |
|--------|----------|------------|--------|
| Dense  | 98.2%    | 1.2M       | 4.8MB  |
| TriX   | 97.8%    | 0.5M       | 0.3MB  |

Conditions: [exact setup]
Code: [link to script]
```

---

## Phase 6: Academic Positioning

### For Paper/Publication

**Title style**: "TriX: Ternary Weights as Content-Addressable
Routing for Efficient Neural Networks"

**Abstract structure**:
1. Problem: Routing overhead in sparse architectures
2. Insight: Weight structure can serve as routing signal
3. Method: Ternary quantization with signature-based dispatch
4. Results: 16x compression, competitive accuracy
5. Contribution: Zero-parameter routing mechanism

**Not in abstract**:
- Philosophical framing
- Experimental meta-learning (separate paper if validated)

### For Course Projects

Clear enough that a grad student can reproduce and extend.
Document the "why" not just the "what."

---

## Implementation Priority

### Must Do (Before Any Release)
1. Language cleanup in guardian/ module
2. README rewrite
3. One working example

### Should Do (For Credibility)
4. Rename guardian → observer
5. Benchmark documentation
6. QUICKSTART.md

### Nice To Have (For Polish)
7. Full tutorial
8. Multiple examples
9. Comparison tables

---

## Success Criteria

The project is accessible when:

1. A hobbyist can get a working example in 10 minutes
2. A researcher reads the docs without cringing
3. A practitioner knows immediately if this fits their use case
4. The experimental parts are clearly marked
5. The core claims are substantiated with evidence

---

## Notes

The goal is not to hide the project's ambitions or origins. It's to
present them in a way that invites engagement rather than dismissal.

The underlying ideas are good. The presentation needs work. That's
a solvable problem.
