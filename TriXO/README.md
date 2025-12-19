# TriX: Ternary Routing with Isomorphic Execution

**A 2-bit sparse neural architecture with zero-parameter emergent routing.**

> *"Don't store what you can XOR."*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-356%20passed-brightgreen.svg)](#testing)

---

## Abstract

TriX introduces a novel neural network architecture where **routing emerges from weight structure** rather than learned gating networks. By constraining weights to ternary values `{-1, 0, +1}`, each computational tile develops a unique signature that enables content-addressable routing with zero additional parameters.

**Key Results:**
- **16× memory compression** via 2-bit weight packing (4 weights per byte)
- **129× signature compression** via XOR superposition (Mesa 13)
- **Sparse computation** - only winning tiles compute per input
- **Zero routing parameters** - routing emerges from weight signatures
- **14% perplexity improvement** on TinyShakespeare with 2.3× fewer parameters
- **Deterministic routing** - bit-exact reproducible decisions

## What's New: Mesa 13 - XOR Superposition Compression

**Mesa 13** delivers 129× signature compression with deterministic O(1) routing.

```python
from trix import HierarchicalTriXFFN

# Train your model
ffn = HierarchicalTriXFFN(d_model=512, num_tiles=64)
# ... training ...

# Compress for inference (128KB → 1KB)
ffn.compress_signatures()
stats = ffn.get_compression_stats()
print(f"Compression: {stats['tile'].compression_ratio:.1f}x")

# Deterministic inference
ffn.eval()
output, routing, _ = ffn(x)  # Bit-exact reproducible
```

**Key Results:**
| Metric | Target | Achieved |
|--------|--------|----------|
| Compression ratio | 11.6× | **129×** |
| Routing determinism | 100% | **100%** |
| Test coverage | Full | **64 tests** |

**How it works:** Trained signatures exhibit ~99% structural similarity. XOR superposition stores one centroid + sparse deltas:

```
For ternary vectors: argmax(dot) = argmin(hamming)
```

This preserves routing decisions exactly while compressing 128KB → 1KB.

See [Mesa 13 Documentation](docs/MESA13_XOR_SUPERPOSITION.md) for full details.

## Mesa 11: Unified Addressing Theory

**Mesa 11** introduces the theoretical foundation explaining why TriX works across diverse domains.

**Core Discovery**: Content-addressing is the universal computational primitive. Temporal addressing (pipelines) and spatial addressing (graphs) are restricted subspaces of a unified address space.

```
Address = [position_dims | topology_dims | feature_dims]

Temporal = [pos, 0, 0]     ← Pipelines, sequences
Spatial  = [0, top, 0]     ← Graphs, recurrence  
Content  = [0, 0, feat]    ← TriX signatures
Mixed    = [pos, top, feat] ← Learned blend
```

**All 8 Validation Experiments CONFIRMED**:

| Experiment | Result | Evidence |
|------------|--------|----------|
| 1. Temporal ⊂ Content | **CONFIRMED** | 0.00 error, 100% accuracy |
| 2. Mixed Addressing | **CONFIRMED** | 95.6% vs baselines |
| 3. Spatial ⊂ Content | **CONFIRMED** | 100% topology preserved |
| 4. Manifold Warping | **CONFIRMED** | 0.077 signature movement |
| 5. Geodesic Tracing | **CONFIRMED** | 100% match all metrics |
| 6. Metric Construction | **CONFIRMED** | 40% route diff by metric |
| 6b. λ-Slider Control | **CONFIRMED** | 100% control, 0 weight updates |
| 7. Curvature & Generalization | **CONFIRMED** | r=+0.712 correlation |

**Geometric Framework**: The unified address space is a literal geometric manifold where:
- Signatures are points, inputs are queries, routing follows geodesics
- Training warps the manifold to align Voronoi cells with task structure
- The metric is a design choice that determines computation
- Smoother manifolds (lower curvature) generalize better
- **Geometry is programmable at inference time** via the λ-slider

> *"Weights tell the Manifold how to curve, Manifold tells the Query how to move."*
> 
> *"We're not in the Grid anymore. We're writing its physics."*

See [Mesa 11 Documentation](docs/MESA11_UAT.md) for full theory and experiments.

## Core Insight

> *"Don't learn what you can read."*

Traditional mixture-of-experts models learn separate routing networks. TriX observes that ternary weights already encode preferences:

```
Weight = +1  →  "I want this feature"
Weight = -1  →  "I want the opposite"  
Weight =  0  →  "I don't care"
```

These preferences form **signatures** that enable routing without additional parameters:

```python
signature = weights.sum(dim=0).sign()  # Tile's "address"
scores = input @ signatures.T          # Content lookup
winner = scores.argmax()               # Route to best match
```

## Installation

**Requirements:** Python ≥3.10, PyTorch ≥2.0, NumPy ≥1.24

```bash
# Clone the repository
git clone https://github.com/your-org/trix.git
cd trix

# Install in development mode
pip install -e .

# Verify installation
python -c "from trix import HierarchicalTriXFFN; print('TriX ready')"
```

## Quick Start

### Drop-in FFN Replacement

```python
import torch
from trix import HierarchicalTriXFFN

# Create input tensor
x = torch.randn(batch_size, seq_len, d_model)

# Replace your FFN with TriX
ffn = HierarchicalTriXFFN(
    d_model=512,
    num_tiles=64,          # More tiles = more specialists
    tiles_per_cluster=8,   # Hierarchical routing granularity
)

# Forward pass returns (output, routing_info, aux_losses)
output, routing_info, aux_losses = ffn(x)

# Training: include auxiliary losses for balanced routing
loss = task_loss + aux_losses['total_aux']
```

### Full Transformer Block

```python
from trix import HierarchicalTriXBlock

block = HierarchicalTriXBlock(
    d_model=512,
    n_heads=8,
    num_tiles=64,
    tiles_per_cluster=8,
)

output, routing_info, aux_losses = block(x)
```

### Sparse Lookup (Routing IS Computation)

```python
from trix import SparseLookupFFN

# Even more aggressive: routing selects direction, spline selects magnitude
ffn = SparseLookupFFN(d_model=512, num_tiles=64)
output, routing_info, aux_losses = ffn(x)
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT TENSOR                              │
│                      (batch, seq, d_model)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNATURE MATCHING                            │
│                                                                  │
│   input ──→ [cluster_sigs] ──→ top_cluster ──→ [tile_sigs] ──→ winner
│                                                                  │
│   Two-level hierarchy: O(√n) routing for n tiles                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     WINNING TILE                                 │
│                                                                  │
│   2-bit ternary weights: W ∈ {-1, 0, +1}^(d_model × d_hidden)   │
│   Sparse computation: only winner executes                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT TENSOR                               │
│                      (batch, seq, d_model)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

| Component | Description | Use Case |
|-----------|-------------|----------|
| `HierarchicalTriXFFN` | 2-level hierarchical routing | Large-scale (64+ tiles) |
| `HierarchicalTriXBlock` | Full transformer block | Drop-in replacement |
| `SparseLookupFFN` | Routing as computation | Maximum sparsity |
| `SparseTriXFFN` | Simple 4-tile FFN | Baseline experiments |
| `TemporalTileLayer` | State-aware routing | Sequential tasks |
| `CompiledDispatch` | O(1) compiled paths | Inference optimization |
| `TriXLinear` | Base ternary linear | Custom architectures |

## Results

### TinyShakespeare Character-Level LM

| Model | Parameters | Val PPL | vs Baseline |
|-------|------------|---------|-------------|
| Sparse-4tiles | — | 19.26 | — |
| Hierarchical-16tiles | 826,304 | 17.16 | −10.9% |
| **SparseLookup-64** | **366,412** | **16.56** | **−14.0%** |

Configuration: d_model=128, n_layers=4, num_tiles=64, tiles_per_cluster=8, seed=42

### Reproduce

```bash
python scripts/benchmark_ffn.py
```

## Project Structure

```
TriXO/
├── src/trix/
│   ├── __init__.py          # Public API exports
│   ├── nn/                   # Neural network modules
│   │   ├── hierarchical.py   # HierarchicalTriXFFN
│   │   ├── xor_superposition.py  # XOR compression (129×)
│   │   ├── sparse_lookup.py  # SparseLookupFFN
│   │   ├── temporal_tiles.py # TemporalTileLayer
│   │   ├── compiled_dispatch.py
│   │   └── ...
│   ├── kernel/               # 2-bit NEON kernel
│   │   ├── bindings.py       # Python bindings
│   │   ├── trix.cpp          # C++ implementation
│   │   └── trix.h
│   ├── qat/                  # Quantization-aware training
│   └── compiler/             # Neural circuit compiler
├── tests/                    # Test suite (268 tests)
├── examples/                 # Usage examples
├── scripts/                  # Benchmarks and validation
└── docs/                     # Documentation
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/trix --cov-report=html
```

Tests validate:
- Forward/backward pass correctness
- Routing behavior and stability
- Weight packing/unpacking
- Hierarchical dispatch
- Temporal state management
- Compiled path optimization

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and routing mechanics
- [Theory](docs/THEORY.md) - Mathematical foundations
- [Mesa 11: Unified Addressing Theory](docs/MESA11_UAT.md) - Why TriX works
- [Mesa 13: XOR Superposition](docs/MESA13_XOR_SUPERPOSITION.md) - 129× signature compression
- [API Reference](docs/API.md) - Complete API documentation
- [Quick Start Tutorial](docs/QUICKSTART.md) - Step-by-step guide
- [Benchmarks](docs/BENCHMARKS.md) - Performance methodology

## Experimental: Mesa 12 (Training Observer)

The `trix.guardian` module contains **experimental research code** for adaptive
training through self-observation. This is not production-ready.

**What it provides:**
- `TrainingObserver`: Monitors training dynamics and can apply bounded interventions
- `ProgrammableTileBank`: Tiles with read/write interface for signature manipulation
- `AdaptiveTrainingPipeline`: 4-phase training (Exploration → Expedition → Convergence → Mastery)

**Current status:**
- Observation infrastructure: Implemented, tested (101 tests)
- Adaptive learning loop: Incomplete, requires validation
- Proof that intervention helps: Not yet demonstrated

See [Mesa 12 Documentation](docs/MESA12.md) for details.

```python
# Experimental usage
from trix.guardian import TrainingObserver, ProgrammableTileBank

tile_bank = ProgrammableTileBank(num_tiles=16, d_model=128, d_hidden=256)
observer = TrainingObserver(d_model=128, num_tiles=16)
# ... see docs for full usage
```

## Hardware Support

| Platform | Status | Notes |
|----------|--------|-------|
| NVIDIA Jetson AGX | ✅ Tested | Primary development target |
| CUDA GPUs | ⚠️ Untested | Should work, PRs welcome |
| CPU | ✅ Tested | Full functionality, no NEON acceleration |
| Apple Silicon | ⚠️ Untested | PyTorch MPS should work |

## Citation

If you use TriX in your research, please cite:

```bibtex
@software{trix2024,
  title = {TriX: Ternary Routing with Isomorphic Execution},
  author = {TriX Contributors},
  year = {2024},
  url = {https://github.com/your-org/trix},
  note = {2-bit sparse neural architecture with emergent routing}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas of interest:
- Hardware backends (CUDA kernels, Metal, TPU)
- New routing strategies
- Benchmark reproductions
- Documentation improvements

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

TriX builds on insights from:
- Mixture of Experts architectures
- Ternary neural networks
- Content-addressable memory systems
- Kolmogorov-Arnold representation theorem

---

**Core Principle:** *Wisdom is knowing when not to compute.*
