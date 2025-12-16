# TriX v2 Quickstart Guide

*From zero to compiled dispatch in 10 minutes.*

---

## What Is TriX?

TriX is a neural network architecture where **routing IS computation**. Instead of dense layers that process everything the same way, TriX routes inputs to specialized "tiles" based on learned signatures.

**Key insight**: Tiles naturally specialize to semantic categories without supervision.

---

## Installation

```python
# From the trix_latest directory
import sys
sys.path.insert(0, '/workspace/trix_latest/src')

from trix.nn import (
    SparseLookupFFNv2,      # The core FFN with surgery + regularizers
    CompiledDispatch,        # Path compilation wrapper
)
```

---

## Quick Example: Basic Usage

```python
import torch
from trix.nn import SparseLookupFFNv2

# Create a TriX FFN
ffn = SparseLookupFFNv2(
    d_model=128,         # Model dimension
    num_tiles=16,        # Number of specialist tiles
    tiles_per_cluster=4, # Tiles per routing cluster
)

# Forward pass
x = torch.randn(2, 32, 128)  # (batch, seq, d_model)
output, routing_info, aux_losses = ffn(x)

# output: transformed tensor, same shape as input
# routing_info: which tiles were selected
# aux_losses: auxiliary losses for training
```

---

## The Five Mesas

TriX v2 provides five levels of capability:

### Mesa 1: Discovery
Tiles specialize to semantic regions automatically.

```python
# After training, check what tiles learned
stats = ffn.get_routing_stats()
print(f"Active tiles: {stats['active_tiles']}")
print(f"Usage distribution: {stats['usage_std']:.3f}")
```

### Mesa 2: Partnership
See what was learned, edit it, improve it.

```python
# See which classes route to which tiles (claim tracking)
output, info, aux = ffn(x, labels=class_labels)
claim_rate = ffn.get_claim_rate(tile_idx=0, target_class=0)
print(f"Tile 0 claims {claim_rate:.0%} of class 0")

# Edit a tile's signature (surgery)
signature = torch.zeros(128)
signature[:16] = 1.0  # Respond to first 16 dims
ffn.insert_signature(tile_idx=0, signature=signature, freeze=True, tag="my_tile")
```

### Mesa 3: Compilation
Freeze routing decisions for O(1) dispatch.

```python
from trix.nn import CompiledDispatch

# Create compiler
compiler = CompiledDispatch(ffn)

# Profile what was learned
profiles = compiler.profile_all(num_classes=10)
for class_id, stats in profiles.items():
    print(f"Class {class_id}: tile={stats.mode_tile}, freq={stats.mode_frequency:.0%}")

# Compile stable classes
compiled = compiler.compile_stable(threshold=0.5)
print(f"Compiled {len(compiled)} classes")

# Use compiled dispatch (O(1) for known classes)
output, info, aux = compiler.forward(x, class_hint=class_id, confidence=0.9)
if info['compiled']:
    print("Used compiled path!")
```

### Mesa 4: Temporal Binding
Route based on (input, state) for temporal patterns.

```python
from trix.nn import TemporalTileLayer

# Create temporal layer
temporal = TemporalTileLayer(
    d_model=32,
    d_state=16,
    num_tiles=8,
)

# Process sequence with state
state = temporal.init_state(batch_size)
for token in sequence:
    output, state, info = temporal(token, state)
    # Tiles learn state transitions - the counter emerges from routing
```

### Mesa 5: Pure TriX FFT
Tiles compute. Routing controls. Everything is TriX.

```python
# The key insight: tiles ARE the operations, routing IS the control
# No external organs - tiles learn ADD and SUB directly
# Routing selects which tile to use

# See experiments/fft_atoms/pure_trix_butterfly.py
# Result: 100% accuracy on butterfly (a,b) → (a+b, a-b)
# Tile specialization: SUM specialist, DIFF specialist
```

---

## Full Training Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from trix.nn import SparseLookupFFNv2, CompiledDispatch

# 1. Create model
class MyModel(nn.Module):
    def __init__(self, d_model=128, num_classes=10):
        super().__init__()
        self.embed = nn.Embedding(1000, d_model)
        self.ffn = SparseLookupFFNv2(
            d_model=d_model,
            num_tiles=16,
            tiles_per_cluster=4,
            ternary_weight=0.01,    # Push signatures to {-1, 0, +1}
            sparsity_weight=0.01,   # Encourage sparse signatures
            diversity_weight=0.01,  # Encourage diverse signatures
        )
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x, labels=None):
        h = self.embed(x)
        h, routing_info, aux_losses = self.ffn(h, labels=labels)
        logits = self.classifier(h.mean(dim=1))
        return logits, routing_info, aux_losses

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 2. Train with claim tracking
for epoch in range(10):
    model.ffn.reset_claim_tracking()
    
    for batch in dataloader:
        x, y = batch
        logits, info, aux = model(x, labels=y)
        
        loss = F.cross_entropy(logits, y) + aux['total_aux']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Check signature quality
    stats = model.ffn.get_island_stats()
    print(f"Epoch {epoch}: ternary={stats['ternary_fraction']:.0%}, "
          f"sparsity={stats['sparsity']:.0%}")

# 3. Compile stable classes
compiler = CompiledDispatch(model.ffn)
compiled = compiler.compile_stable(threshold=0.5)
print(f"Compiled {len(compiled)} classes")

# 4. Use in inference
model.eval()
with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        h = model.embed(x)
        
        # Use compiled dispatch
        h, info, _ = compiler.forward(h, class_hint=y[0].item(), confidence=0.9)
        
        logits = model.classifier(h.mean(dim=1))
```

---

## Surgery API Reference

### Insert a Signature
```python
# Design a signature
sig = torch.zeros(d_model)
sig[:16] = 1.0   # Positive on first 16 dims
sig[16:24] = -1.0  # Negative on next 8

# Insert and freeze
ffn.insert_signature(tile_idx=0, signature=sig, freeze=True, tag="my_concept")
```

### Freeze/Unfreeze
```python
ffn.freeze_signature(tile_idx=0)   # Lock (no gradient updates)
ffn.unfreeze_signature(tile_idx=0) # Unlock (allow learning)
ffn.is_frozen(tile_idx=0)          # Check status
```

### Analyze
```python
analysis = ffn.get_signature_analysis(tile_idx=0)
# Returns:
#   positive_dims: list of dims with +1
#   negative_dims: list of dims with -1
#   zero_count: number of zero dims
#   frozen: bool

history = ffn.get_surgery_history()
# Returns list of all surgery operations
```

### Claim Tracking
```python
# Enable during forward
output, info, aux = ffn(x, labels=class_labels)

# Query
claim_rate = ffn.get_claim_rate(tile_idx=0, target_class=0)
dominant_class, purity = ffn.get_tile_purity(tile_idx=0)

# Reset
ffn.reset_claim_tracking()
```

---

## CompiledDispatch API Reference

### Lifecycle
```python
compiler = CompiledDispatch(ffn)

# Profile
profiles = compiler.profile_all(num_classes=10)
stats = compiler.profile(class_id=0)

# Compile
compiled = compiler.compile_stable(threshold=0.5)
entry = compiler.compile(class_id=0, tile_idx=2)

# Execute
output, info, aux = compiler.forward(x, class_hint=0, confidence=0.9)

# Monitor
stats = compiler.get_stats()
drifted = compiler.check_drift(threshold=0.3)
compiler.recompile_drifted()

# Serialize
table = compiler.export_dispatch_table()
compiler.import_dispatch_table(table)
```

### Stats
```python
stats = compiler.get_stats()
# Returns:
#   compiled_hits: times compiled path used
#   compiled_misses: times guard failed
#   dynamic_calls: times dynamic routing used
#   hit_rate: compiled_hits / total
#   num_compiled_classes: how many classes compiled
```

---

## Regularizers

### Ternary Regularizer
Pushes signatures toward {-1, 0, +1}. Makes routing decisions crisp.

```python
ffn = SparseLookupFFNv2(..., ternary_weight=0.01)
```

### Sparsity Regularizer
Encourages sparse signatures (many zeros). Each tile responds to few dimensions.

```python
ffn = SparseLookupFFNv2(..., sparsity_weight=0.01)
```

### Diversity Regularizer
Penalizes similar signatures. Tiles become distinct.

```python
ffn = SparseLookupFFNv2(..., diversity_weight=0.01)
```

### Check Quality
```python
stats = ffn.get_island_stats()
print(f"Ternary: {stats['ternary_fraction']:.0%}")
print(f"Sparsity: {stats['sparsity']:.0%}")
print(f"Diversity: {stats['diversity']:.2f}")
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Just v2 tests
pytest tests/test_sparse_lookup_v2.py -v

# Just compiled dispatch tests
pytest tests/test_compiled_dispatch.py -v

# A/B harness tests
pytest tests/test_ab_harness.py -v
```

---

## Running Experiments

```bash
# 6502 organ discovery
python experiments/trix_6502_v2_organs.py

# A/B comparison (dynamic vs compiled)
python experiments/ab_harness_compiled.py

# v2 benchmark
python experiments/benchmark_v2_rigorous.py
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/trix/nn/sparse_lookup_v2.py` | Core FFN with surgery + regularizers |
| `src/trix/nn/compiled_dispatch.py` | Path compilation |
| `tests/test_sparse_lookup_v2.py` | 39 v2 tests |
| `tests/test_compiled_dispatch.py` | 21 compilation tests |
| `tests/test_ab_harness.py` | A/B comparison tests |
| `docs/SPARSE_LOOKUP_V2_API.md` | Full API reference |
| `docs/SESSION_SUMMARY_MESA_1_2_3.md` | Complete session docs |

---

## Common Patterns

### Pattern 1: Train → Compile → Deploy
```python
# Train
for epoch in range(100):
    train_epoch(model)

# Compile
compiler = CompiledDispatch(model.ffn)
compiler.compile_stable(threshold=0.5)

# Deploy
save(compiler.export_dispatch_table())
```

### Pattern 2: Monitor → Recompile
```python
# In production
while serving:
    output = compiler.forward(x, class_hint=c)
    
    if time_for_check():
        drifted = compiler.check_drift()
        if drifted:
            compiler.recompile_drifted()
```

### Pattern 3: Surgery for Known Concepts
```python
# You know what "math" looks like
math_sig = design_math_signature()
ffn.insert_signature(tile_idx=0, signature=math_sig, freeze=True, tag="math")

# Train - tile 0 will claim math inputs
train(model)

# Verify
claim_rate = ffn.get_claim_rate(tile_idx=0, target_class=MATH_CLASS)
print(f"Math tile claims {claim_rate:.0%} of math")
```

---

## Next Steps

1. **Read the full API docs**: `docs/SPARSE_LOOKUP_V2_API.md`
2. **Understand the theory**: `docs/SEMANTIC_GEOMETRY_THESIS.md`
3. **See what we proved**: `docs/SESSION_SUMMARY_MESA_1_2_3.md`
4. **Run experiments**: `experiments/` directory
5. **Explore the notes**: `notes/` for raw thinking process

---

## Philosophy

> "TriX is the cartographer, not the compute."

TriX discovers WHERE computation should happen (which tiles handle which inputs). The actual computation is done by the tiles. This separation enables:

- **Interpretability**: See what each tile learned
- **Editability**: Fix tiles that learned wrong
- **Compilation**: Freeze routing for speed
- **Composition**: Assemble tiles into systems

The goal isn't just accuracy. It's **operability** - turning neural networks from things that behave into things that can be operated.

---

*Welcome to TriX. The routing is the computation.*
