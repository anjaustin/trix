# SparseLookupFFNv2 API Documentation

*Complete API reference for the enhanced SparseLookupFFN with surgery and regularization.*

---

## Overview

`SparseLookupFFNv2` extends `SparseLookupFFN` with:
- **Signature Surgery API**: Insert, freeze, unfreeze, analyze signatures
- **Island Regularizers**: Ternary, sparsity, diversity losses
- **Score Calibration**: Spline-based routing score calibration
- **Claim Tracking**: Track which classes route to which tiles

---

## Installation

```python
from trix.nn import SparseLookupFFNv2, SparseLookupBlockV2
```

---

## Basic Usage

```python
import torch
from trix.nn import SparseLookupFFNv2

# Create model
ffn = SparseLookupFFNv2(
    d_model=128,
    num_tiles=64,
    tiles_per_cluster=8,
    ternary_weight=0.01,    # Island regularizers
    sparsity_weight=0.01,
    diversity_weight=0.01,
)

# Forward pass
x = torch.randn(2, 32, 128)
output, routing_info, aux_losses = ffn(x)

# Training
loss = task_loss + aux_losses['total_aux']
loss.backward()
```

---

## Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `d_model` | int | 128 | Model dimension |
| `num_tiles` | int | 64 | Number of specialist tiles |
| `tiles_per_cluster` | int | 8 | Tiles per routing cluster |
| `grid_size` | int | 16 | Magnitude spline grid size |
| `compress_hidden` | int | d_model//4 | Compression network hidden dim |
| `dropout` | float | 0.1 | Dropout rate |
| `use_score_calibration` | bool | True | Enable score calibration spline |
| `ternary_weight` | float | 0.01 | Weight for ternary regularizer |
| `sparsity_weight` | float | 0.01 | Weight for sparsity regularizer |
| `diversity_weight` | float | 0.01 | Weight for diversity regularizer |

---

## Signature Surgery API

### Insert a Signature

```python
# Design a signature
signature = torch.zeros(d_model)
signature[:16] = 1.0   # Positive on first 16 dims
signature[16:24] = -1.0  # Negative on next 8 dims

# Insert into tile 0, frozen
ffn.insert_signature(
    tile_idx=0,
    signature=signature,
    freeze=True,
    tag="my_concept"
)
```

### Freeze/Unfreeze

```python
# Freeze a signature (no gradient updates)
ffn.freeze_signature(tile_idx=0)

# Check if frozen
if ffn.is_frozen(0):
    print("Tile 0 is frozen")

# Unfreeze (allow learning)
ffn.unfreeze_signature(tile_idx=0)
```

### Analyze Signatures

```python
# Get signature analysis
analysis = ffn.get_signature_analysis(tile_idx=0)
# Returns:
# {
#     'tile_idx': 0,
#     'positive_dims': [0, 1, 2, ...],
#     'negative_dims': [16, 17, ...],
#     'zero_count': 40,
#     'frozen': True
# }

# Get surgery history
history = ffn.get_surgery_history()
# Returns list of all surgery operations
```

---

## Island Regularizers

### Ternary Regularizer

Encourages signatures toward {-1, 0, +1}.

```python
# Computed automatically in forward pass
aux_losses['ternary']  # Weighted ternary loss

# Manual computation
ternary_loss = ffn.compute_ternary_loss()
```

### Sparsity Regularizer

Encourages sparse signatures (many zeros).

```python
aux_losses['sparsity']  # Weighted sparsity loss

# Manual
sparsity_loss = ffn.compute_sparsity_loss()
```

### Diversity Regularizer

Penalizes similar signatures.

```python
aux_losses['diversity']  # Weighted diversity loss

# Manual
diversity_loss = ffn.compute_diversity_loss()
```

### Using in Training

```python
output, routing_info, aux_losses = ffn(x)

# aux_losses contains:
# - 'balance': Load balancing loss
# - 'ternary': Ternary regularizer
# - 'sparsity': Sparsity regularizer
# - 'diversity': Diversity regularizer
# - 'total_aux': Sum of all

task_loss = compute_task_loss(output, targets)
total_loss = task_loss + aux_losses['total_aux']
total_loss.backward()
```

---

## Claim Tracking

Track which classes route to which tiles.

### Enable Tracking

```python
# Pass labels during forward
labels = torch.tensor([[0, 1, 2, ...]])  # Class labels
output, routing_info, aux_losses = ffn(x, labels=labels)
```

### Query Claims

```python
# Get claim rate: fraction of class C routed to tile T
claim_rate = ffn.get_claim_rate(tile_idx=0, target_class=0)
# Returns: float in [0, 1]

# Get tile purity
dominant_class, purity = ffn.get_tile_purity(tile_idx=0)
# dominant_class: which class routes most to this tile
# purity: fraction of tile's traffic from dominant class

# Reset tracking
ffn.reset_claim_tracking()
```

---

## Statistics

### Island Stats

```python
stats = ffn.get_island_stats()
# Returns:
# {
#     'ternary_fraction': 0.95,  # Fraction of values near {-1,0,+1}
#     'sparsity': 0.70,          # Fraction of near-zero values
#     'mean_pairwise_similarity': 0.05,
#     'diversity': 0.95,         # 1 - mean_similarity
# }
```

### Routing Stats

```python
stats = ffn.get_routing_stats()
# Returns:
# {
#     'num_tiles': 64,
#     'num_clusters': 8,
#     'active_tiles': 62,
#     'frozen_tiles': 2,
#     'usage_mean': 0.016,
#     'usage_std': 0.008,
# }
```

### Reset Stats

```python
ffn.reset_stats()  # Reset usage and claim tracking
```

---

## SparseLookupBlockV2

Transformer block with SparseLookupFFNv2.

```python
from trix.nn import SparseLookupBlockV2

block = SparseLookupBlockV2(
    d_model=128,
    n_heads=4,
    num_tiles=64,
    tiles_per_cluster=8,
    ternary_weight=0.01,
    sparsity_weight=0.01,
    diversity_weight=0.01,
)

x = torch.randn(2, 32, 128)
output, routing_info, aux_losses = block(x, is_causal=True)

# Access FFN surgery API
block.ffn.insert_signature(0, signature, freeze=True)
```

---

## Score Calibration Spline

Calibrates raw routing scores to gate values.

```python
from trix.nn.sparse_lookup_v2 import ScoreCalibrationSpline

spline = ScoreCalibrationSpline(num_knots=8)

scores = torch.randn(32, 64)  # Raw scores
gates = spline(scores)         # Calibrated [0, 1]
```

---

## Complete Example: Surgery Workflow

```python
import torch
from trix.nn import SparseLookupFFNv2

# 1. Create model
ffn = SparseLookupFFNv2(
    d_model=64,
    num_tiles=16,
    tiles_per_cluster=4,
    ternary_weight=0.01,
)

# 2. Train normally for a while
optimizer = torch.optim.Adam(ffn.parameters())
for _ in range(100):
    x = torch.randn(8, 32, 64)
    output, _, aux = ffn(x)
    loss = output.sum() + aux['total_aux']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 3. Insert a designed signature for a specific concept
concept_signature = torch.zeros(64)
concept_signature[:8] = 1.0    # Positive features
concept_signature[8:12] = -1.0  # Negative features

ffn.insert_signature(0, concept_signature, freeze=True, tag="my_concept")

# 4. Continue training (signature stays fixed)
for _ in range(100):
    x = torch.randn(8, 32, 64)
    labels = torch.randint(0, 10, (8, 32))
    output, _, aux = ffn(x, labels=labels)
    loss = output.sum() + aux['total_aux']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. Check if tile claims target class
claim_rate = ffn.get_claim_rate(tile_idx=0, target_class=0)
print(f"Tile 0 claims {claim_rate:.1%} of class 0")

# 6. Optionally unfreeze to let it refine
ffn.unfreeze_signature(0)

# 7. Check signature evolution
analysis = ffn.get_signature_analysis(0)
print(f"Final positive dims: {analysis['positive_dims']}")
```

---

## Test Suite

39 tests covering all functionality:

```bash
pytest tests/test_sparse_lookup_v2.py -v
```

Test categories:
- Basic functionality (6 tests)
- Surgery API (8 tests)
- Regularizers (8 tests)
- Score calibration (4 tests)
- Block integration (4 tests)
- Island stats (3 tests)
- Edge cases (6 tests)

---

## Benchmark Results

From rigorous benchmark (December 15, 2024):

| Test | Result |
|------|--------|
| Basic forward/backward | ✓ PASS |
| Surgery insert | ✓ PASS |
| Surgery freeze/unfreeze | ✓ PASS |
| Frozen signature stability | ✓ PASS (max diff: 0.0000) |
| Ternary regularizer | ✓ PASS (prefers ternary) |
| Sparsity regularizer | ✓ PASS (prefers sparse) |
| Diversity regularizer | ✓ PASS (prefers diverse) |
| Training dynamics | ✓ PASS (sparsity 46%→70%) |
| Claim tracking | ✓ PASS (80% claim rate) |
| Block integration | ✓ PASS |
| Edge cases | ✓ PASS |

---

## Files

- `src/trix/nn/sparse_lookup_v2.py` — Implementation
- `tests/test_sparse_lookup_v2.py` — Test suite (39 tests)
- `experiments/benchmark_v2_rigorous.py` — Benchmark script

---

*SparseLookupFFNv2: Where routing is computation, and addresses are writable.*
