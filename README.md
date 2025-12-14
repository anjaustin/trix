# TriX

**A 2-Bit Conditional Ternary Neural Architecture with Learned Computational Sparsity and Emergent Routing**

---

## What is TriX?

TriX is a drop-in replacement for transformer FFN layers that provides:

- **16x memory compression** (true 2-bit weights, 4 per byte)
- **4x inference speedup** (sparse routing, only 1 tile computes per input)
- **Zero routing parameters** (routing emerges from weight structure)
- **13.4% quality improvement** over baseline on language modeling

NOTE: This repo will work as is on a Jeston AGX Thor. It has not been compiled and tested on any other hardware.

## Quick Start

```bash
pip install torch numpy
pip install -e .
```

```python
from trix import HierarchicalTriXFFN

# Replace your FFN
ffn = HierarchicalTriXFFN(
    d_model=512,
    num_tiles=16,          # More tiles = more specialists
    tiles_per_cluster=4,
)

# Use like any PyTorch module
output, routing_info, aux_losses = ffn(x)

# Training: add aux_losses to your loss
loss = task_loss + aux_losses['total_aux']
```

See `examples/nvidia_quickstart.py` for complete examples.

---

## How It Works

### 1. Ternary Weights = Votes

Each weight is {-1, 0, +1}:
- `+1` = "I want this feature"
- `-1` = "I want the opposite"  
- `0` = "I don't care"

### 2. Signatures = Addresses

Each tile's signature is derived from its weights:
```python
signature = weights.sum(dim=0).sign()
```

### 3. Routing = Content Lookup

Inputs route to the tile whose signature best matches:
```python
scores = input @ signatures.T
winner = scores.argmax()
```

**Zero learned parameters. Three lines of code.**

---

## Key Components

| Component | Use Case |
|-----------|----------|
| `HierarchicalTriXFFN` | FFN with hierarchical routing (recommended) |
| `HierarchicalTriXBlock` | Full transformer block |
| `SparseTriXFFN` | Simple 4-tile FFN |
| `TriXLinear` | Low-level ternary linear layer |

---

## Validated Results

| Model | Val PPL | vs Baseline |
|-------|---------|-------------|
| Sparse-4tiles | 19.26 | — |
| **Hierarchical-16tiles** | **16.67** | **-13.4%** |

Tested on TinyShakespeare character-level language modeling.

---

## Architecture

```
Input
  │
  ▼
┌─────────────────────────────┐
│  Input Normalization        │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  Cluster Routing (Level 1)  │  ← O(num_clusters) comparisons
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  Tile Routing (Level 2)     │  ← O(tiles_per_cluster) comparisons
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  Tile Computation (2-bit)   │  ← Only winning tile computes
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  Residual Connection        │  ← output = input + tile_output
└─────────────────────────────┘
  │
  ▼
Output
```

---

## Project Structure

```
trix/
├── src/trix/
│   ├── nn/
│   │   ├── hierarchical.py  # HierarchicalTriXFFN (recommended)
│   │   ├── sparse.py        # SparseTriXFFN (simple)
│   │   └── trix.py          # TriXFFN (classic)
│   ├── kernel/              # 2-bit NEON kernel
│   └── qat/                 # Quantization-aware training
├── tests/                   # Test suite
├── examples/                # Usage examples
└── docs/                    # Documentation
```

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/BUILD_LOG.md` | Complete development journey |
| `docs/ABSTRACT.md` | Technical abstract |
| `docs/BIG_LEAP_SPEC.md` | Hierarchical architecture specification |
| `examples/nvidia_quickstart.py` | Plug-and-play examples |

---

## Core Principle

> **Don't learn what you can read.**

Ternary weights encode preferences.  
Preferences enable routing.  
Routing enables sparsity.  
Sparsity enables speed.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy

No other dependencies.

---

## License

MIT License. See [LICENSE](LICENSE).
