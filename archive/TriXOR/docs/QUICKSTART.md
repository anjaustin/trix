# TriX Quick Start Guide

A hands-on tutorial for researchers getting started with TriX.

## Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Basic familiarity with transformer architectures

## Installation

```bash
git clone https://github.com/your-org/trix.git
cd trix
pip install -e .
```

Verify:
```bash
python -c "from trix import HierarchicalTriXFFN; print('Ready!')"
```

---

## Tutorial 1: Your First TriX Layer

### Step 1: Create a Simple FFN

```python
import torch
from trix import HierarchicalTriXFFN

# Configuration
d_model = 128
num_tiles = 16
batch_size = 4
seq_len = 32

# Create the FFN
ffn = HierarchicalTriXFFN(
    d_model=d_model,
    num_tiles=num_tiles,
    tiles_per_cluster=4,
)

print(f"Parameters: {sum(p.numel() for p in ffn.parameters()):,}")
```

### Step 2: Forward Pass

```python
# Random input
x = torch.randn(batch_size, seq_len, d_model)

# Forward pass
output, routing_info, aux_losses = ffn(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Aux loss:     {aux_losses['total_aux'].item():.4f}")
```

### Step 3: Examine Routing

```python
# Which tiles were selected?
tile_indices = routing_info['global_indices']
print(f"Tile indices shape: {tile_indices.shape}")
print(f"Unique tiles used:  {tile_indices.unique().tolist()}")

# Routing distribution
tile_counts = torch.bincount(tile_indices.flatten(), minlength=num_tiles)
print(f"Tile usage: {tile_counts.tolist()}")
```

---

## Tutorial 2: Training Loop

### Setup

```python
import torch
import torch.nn as nn
from trix import HierarchicalTriXFFN

# Simple model: embedding + TriX FFN + output
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_tiles):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.ffn = HierarchicalTriXFFN(d_model, num_tiles)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        h = self.embed(x)
        h, routing_info, aux_losses = self.ffn(h)
        logits = self.output(h)
        return logits, aux_losses

model = SimpleModel(vocab_size=1000, d_model=128, num_tiles=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### Training Step

```python
def train_step(model, batch, targets, optimizer):
    optimizer.zero_grad()
    
    # Forward
    logits, aux_losses = model(batch)
    
    # Task loss
    task_loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )
    
    # Total loss includes auxiliary losses for balanced routing
    total_loss = task_loss + aux_losses['total_aux']
    
    # Backward
    total_loss.backward()
    optimizer.step()
    
    return {
        'task_loss': task_loss.item(),
        'aux_loss': aux_losses['total_aux'].item(),
        'total_loss': total_loss.item(),
    }
```

### Training Loop

```python
# Dummy data
for epoch in range(10):
    batch = torch.randint(0, 1000, (32, 64))
    targets = torch.randint(0, 1000, (32, 64))
    
    metrics = train_step(model, batch, targets, optimizer)
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: loss={metrics['total_loss']:.4f}")
```

---

## Tutorial 3: Inspecting Signatures

### View Tile Signatures

```python
from trix import HierarchicalTriXFFN

ffn = HierarchicalTriXFFN(d_model=64, num_tiles=8, tiles_per_cluster=4)

# Get all signatures
signatures = ffn.get_signatures()
print(f"Signatures shape: {signatures.shape}")  # (8, 64)

# Each signature is ternary
print(f"Unique values: {signatures.unique().tolist()}")  # [-1, 0, 1]
```

### Visualize Signature Diversity

```python
import matplotlib.pyplot as plt

# Compute pairwise distances
def signature_distance(s1, s2):
    return (s1 != s2).float().sum()

n_tiles = signatures.shape[0]
distances = torch.zeros(n_tiles, n_tiles)
for i in range(n_tiles):
    for j in range(n_tiles):
        distances[i, j] = signature_distance(signatures[i], signatures[j])

plt.imshow(distances, cmap='viridis')
plt.colorbar(label='Hamming Distance')
plt.title('Signature Diversity Matrix')
plt.xlabel('Tile')
plt.ylabel('Tile')
plt.savefig('signature_diversity.png')
```

---

## Tutorial 4: SparseLookup Architecture

### Concept

SparseLookup takes routing further: the routing decision **is** the computation.

```python
from trix import SparseLookupFFN

ffn = SparseLookupFFN(
    d_model=128,
    num_tiles=64,
    spline_knots=8,  # Magnitude modulation
)

x = torch.randn(4, 32, 128)
output, routing_info, aux_losses = ffn(x)

# Check parameter count
params = sum(p.numel() for p in ffn.parameters())
print(f"SparseLookup params: {params:,}")
```

### How It Works

1. **Routing** selects a direction (tile signature)
2. **Spline** modulates magnitude based on input
3. **No matrix multiply** in the hot path

---

## Tutorial 5: Temporal Routing

### For Sequential Tasks

```python
from trix.nn import TemporalTileLayer

temporal = TemporalTileLayer(
    d_model=64,
    d_state=16,  # State dimension
    num_tiles=8,
)

# Process a sequence
x = torch.randn(4, 100, 64)  # (batch, seq, d_model)
output, final_state, routing_infos = temporal.forward_sequence(x)

print(f"Output shape: {output.shape}")
print(f"Final state shape: {final_state.shape}")
print(f"Routing decisions: {len(routing_infos)}")
```

### Stateful Processing

```python
# Process step by step with explicit state
state = temporal.init_state(batch_size=4)

for t in range(10):
    x_t = torch.randn(4, 64)  # Single timestep
    out_t, state, info = temporal(x_t, state)
    print(f"Step {t}: tile={info['tile_indices'][0].item()}")
```

---

## Tutorial 6: Compiled Dispatch

### For Production Inference

```python
from trix.nn import SparseLookupFFNv2, CompiledDispatch

# Train a model (simplified)
ffn = SparseLookupFFNv2(d_model=128, num_tiles=32)

# Wrap with compiled dispatch
compiled = CompiledDispatch(ffn)

# Profile routing for known classes
for class_id in range(10):
    samples = torch.randn(100, 128)  # Representative samples
    stats = compiled.profile_class(class_id, samples)
    print(f"Class {class_id}: dominant_tile={stats.dominant_tile}, "
          f"stability={stats.stability:.2f}")

# Compile stable routes
n_compiled = compiled.compile_stable(threshold=0.9)
print(f"Compiled {n_compiled} classes")

# Inference with hints
x = torch.randn(1, 32, 128)
output, _, _ = compiled(x, class_hint=5, confidence=0.95)
```

---

## Tutorial 7: Weight Packing

### Memory Compression

```python
from trix import pack_weights, unpack_weights

# Get weights from a trained model
ffn = HierarchicalTriXFFN(d_model=256, num_tiles=16)

# Pack all weights
packed = ffn.pack_weights()

# Compare sizes
original_size = sum(p.numel() * 4 for p in ffn.parameters())  # FP32
packed_size = sum(p.numel() for p in packed.values())

print(f"Original: {original_size:,} bytes")
print(f"Packed:   {packed_size:,} bytes")
print(f"Compression: {original_size / packed_size:.1f}x")
```

### Save and Load

```python
# Save packed weights
torch.save(packed, 'model_packed.pt')

# Load and unpack
loaded = torch.load('model_packed.pt')
ffn.unpack_weights(loaded)
```

---

## Common Patterns

### Pattern 1: Replace FFN in Existing Model

```python
# Before
class OldBlock(nn.Module):
    def __init__(self, d_model):
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

# After
class NewBlock(nn.Module):
    def __init__(self, d_model):
        self.ffn = HierarchicalTriXFFN(d_model, num_tiles=64)
    
    def forward(self, x):
        out, _, aux = self.ffn(x)
        return out, aux  # Return aux for training
```

### Pattern 2: Auxiliary Loss Weighting

```python
# Early training: high aux weight for routing stability
# Late training: low aux weight for task focus

def get_aux_weight(epoch, total_epochs):
    # Linear decay from 0.1 to 0.001
    return 0.1 * (1 - epoch / total_epochs) + 0.001

# In training loop
aux_weight = get_aux_weight(epoch, total_epochs)
loss = task_loss + aux_weight * aux_losses['total_aux']
```

### Pattern 3: Monitoring Routing Health

```python
def check_routing_health(ffn, dataloader):
    tile_counts = torch.zeros(ffn.num_tiles)
    
    for batch in dataloader:
        _, routing_info, _ = ffn(batch)
        indices = routing_info['global_indices'].flatten()
        tile_counts += torch.bincount(indices, minlength=ffn.num_tiles)
    
    # Check for dead tiles
    dead_tiles = (tile_counts == 0).sum().item()
    
    # Check for dominant tiles
    max_share = tile_counts.max() / tile_counts.sum()
    
    print(f"Dead tiles: {dead_tiles}/{ffn.num_tiles}")
    print(f"Max tile share: {max_share:.1%}")
    
    return dead_tiles == 0 and max_share < 0.5
```

---

## Next Steps

1. **[Architecture Guide](ARCHITECTURE.md)** - Deep dive into system design
2. **[Theory](THEORY.md)** - Mathematical foundations
3. **[API Reference](API.md)** - Complete API documentation
4. **[Benchmarks](BENCHMARKS.md)** - Reproduce our results
