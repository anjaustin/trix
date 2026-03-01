# TriX Architecture Guide

This document provides a comprehensive overview of the TriX architecture for researchers and developers.

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Core Components](#core-components)
3. [Routing Mechanism](#routing-mechanism)
4. [Hierarchical Organization](#hierarchical-organization)
5. [Weight Representation](#weight-representation)
6. [Training Dynamics](#training-dynamics)
7. [Inference Optimization](#inference-optimization)

---

## Design Philosophy

### The Central Thesis

Traditional neural networks learn dense transformations where every weight participates in every computation. Mixture-of-Experts (MoE) architectures address this by routing inputs to specialized sub-networks, but they require **learned routing networks** with additional parameters.

TriX proposes a third path: **emergent routing** where the routing decision arises naturally from the structure of the weights themselves.

### Key Principles

1. **Ternary Constraints Enable Structure**
   - Weights restricted to {-1, 0, +1} develop interpretable signatures
   - Signatures encode what features each tile "cares about"

2. **Routing Without Routing Networks**
   - No learned gating mechanism
   - Content-addressable lookup via signature matching
   - Zero additional parameters for routing

3. **Sparsity Through Selection**
   - Only the winning tile computes
   - Computation proportional to 1/num_tiles
   - Memory bandwidth reduced via 2-bit packing

---

## Core Components

### TriXLinear

The foundational layer implementing ternary linear transformation:

```python
class TriXLinear(nn.Module):
    """
    Linear layer with ternary weights {-1, 0, +1}.
    
    During training: continuous weights quantized via STE
    During inference: true 2-bit computation
    """
    def __init__(self, in_features, out_features):
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
    def forward(self, x):
        # Straight-through estimator: gradient flows through sign()
        w_ternary = STESign.apply(self.weight)
        return F.linear(x, w_ternary)
```

### TriXTile

A single specialist unit:

```python
class TriXTile(nn.Module):
    """
    One specialist in the mixture.
    
    Components:
    - up_proj: d_model → d_hidden (expansion)
    - down_proj: d_hidden → d_model (compression)
    - signature: derived from weight structure
    """
    def __init__(self, d_model, d_hidden):
        self.up_proj = TriXLinear(d_model, d_hidden)
        self.down_proj = TriXLinear(d_hidden, d_model)
        
    @property
    def signature(self):
        # Tile's "address" in content space
        return self.up_proj.weight.sum(dim=0).sign()
```

### HierarchicalTriXFFN

The main feed-forward network with hierarchical routing:

```
Input (B, T, D)
      │
      ▼
┌─────────────────┐
│ Cluster Routing │  ← O(√n) signature comparisons
│  (8 clusters)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tile Routing   │  ← O(√n) within-cluster comparisons  
│ (8 tiles/cluster)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Winning Tile   │  ← Only 1 of 64 tiles executes
│   Computation   │
└────────┬────────┘
         │
         ▼
Output (B, T, D)
```

---

## Routing Mechanism

### Signature Computation

Each tile's signature is computed from its weights:

```python
def compute_signature(tile):
    """
    Aggregate weight preferences into a signature vector.
    
    Intuition: If many weights for feature i are +1, 
    the tile "wants" high values of feature i.
    """
    # Sum across output dimension
    raw_sig = tile.up_proj.weight.sum(dim=0)  # (d_model,)
    
    # Ternarize to get clean signature
    signature = raw_sig.sign()  # ∈ {-1, 0, +1}^d_model
    
    return signature
```

### Content-Addressable Lookup

Routing is a simple dot product:

```python
def route(input, signatures):
    """
    Route input to best-matching tile.
    
    High score = input aligns with tile's preferences
    """
    # Compute alignment scores
    scores = input @ signatures.T  # (batch, seq, num_tiles)
    
    # Winner-take-all
    winner_idx = scores.argmax(dim=-1)
    
    return winner_idx
```

### Why This Works

Consider a tile with signature `[+1, +1, -1, 0, ...]`:
- Scores high when input has: high feature 0, high feature 1, low feature 2
- Scores low for inputs with opposite pattern
- Ignores features where signature is 0

This creates **natural specialization**: tiles become experts for input subspaces that align with their signatures.

---

## Hierarchical Organization

### The Scaling Problem

With n tiles, naive routing requires O(n) comparisons per token. For n=1000 tiles, this becomes expensive.

### Two-Level Hierarchy

TriX organizes tiles into clusters:

```
64 tiles = 8 clusters × 8 tiles/cluster

Routing cost: O(8) + O(8) = O(16) vs O(64)
General: O(√n) + O(√n) = O(√n) vs O(n)
```

### Cluster Signatures

Each cluster has a representative signature:

```python
def compute_cluster_signature(cluster_tiles):
    """Average signature of tiles in cluster."""
    tile_sigs = [tile.signature for tile in cluster_tiles]
    return torch.stack(tile_sigs).mean(dim=0).sign()
```

### Routing Process

```python
def hierarchical_route(input, clusters):
    # Level 1: Find best cluster
    cluster_sigs = [c.signature for c in clusters]
    cluster_scores = input @ torch.stack(cluster_sigs).T
    best_cluster = cluster_scores.argmax(dim=-1)
    
    # Level 2: Find best tile within cluster
    cluster = clusters[best_cluster]
    tile_sigs = [t.signature for t in cluster.tiles]
    tile_scores = input @ torch.stack(tile_sigs).T
    best_tile = tile_scores.argmax(dim=-1)
    
    return best_cluster, best_tile
```

---

## Weight Representation

### Ternary Encoding

Weights are stored as 2-bit values:

| Value | Encoding |
|-------|----------|
| -1    | 00       |
|  0    | 01       |
| +1    | 10       |
| (unused) | 11    |

### Packing

Four weights pack into one byte:

```python
def pack_weights(weights):
    """Pack ternary weights to 2-bit representation."""
    # Map {-1, 0, +1} to {0, 1, 2}
    encoded = (weights + 1).to(torch.uint8)
    
    # Pack 4 values per byte
    packed = (encoded[0::4] << 6) | (encoded[1::4] << 4) | \
             (encoded[2::4] << 2) | encoded[3::4]
    
    return packed
```

### Memory Savings

| Representation | Bits/Weight | Compression |
|----------------|-------------|-------------|
| FP32           | 32          | 1×          |
| FP16           | 16          | 2×          |
| INT8           | 8           | 4×          |
| **TriX (2-bit)** | **2**     | **16×**     |

---

## Training Dynamics

### Straight-Through Estimator (STE)

The sign() function has zero gradient almost everywhere. STE provides a surrogate gradient:

```python
class STESign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.sign()
    
    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient through as if sign() were identity
        return grad_output
```

### Auxiliary Losses

Balanced routing requires regularization:

```python
def compute_aux_losses(routing_probs):
    # Load balancing: encourage uniform tile usage
    load = routing_probs.mean(dim=[0, 1])  # (num_tiles,)
    target = 1.0 / num_tiles
    load_loss = ((load - target) ** 2).sum()
    
    # Entropy: encourage confident routing
    entropy = -(routing_probs * routing_probs.log()).sum(dim=-1).mean()
    entropy_loss = entropy  # Minimize entropy = maximize confidence
    
    return load_loss + 0.01 * entropy_loss
```

### Signature Evolution

During training, signatures evolve to cover the input space:

```
Epoch 0:   Random signatures, random routing
Epoch 10:  Signatures begin differentiating  
Epoch 50:  Clear specialization emerges
Epoch 100: Stable, diverse signatures
```

---

## Inference Optimization

### Compiled Dispatch

For known input classes, routes can be precomputed:

```python
class CompiledDispatch:
    """O(1) routing via lookup table."""
    
    def compile(self, class_id, representative_input):
        # Compute route once
        route = self.router(representative_input)
        self.lookup_table[class_id] = route
    
    def forward(self, x, class_id):
        if class_id in self.lookup_table:
            # O(1) lookup
            tile_idx = self.lookup_table[class_id]
            return self.tiles[tile_idx](x)
        else:
            # Fallback to dynamic routing
            return self.dynamic_route(x)
```

### NEON Kernel

For ARM platforms, a NEON-accelerated kernel provides:
- Vectorized 2-bit unpacking
- Ternary multiply-accumulate
- ~4× speedup on Jetson platforms

```cpp
// Ternary MAC: accumulate += input * weight
// where weight ∈ {-1, 0, +1}
int8x16_t trix_mac(int8x16_t acc, int8x16_t input, uint8x16_t packed_weights) {
    // Unpack 2-bit weights to int8
    int8x16_t weights = unpack_ternary(packed_weights);
    
    // Ternary multiply: -input, 0, or +input
    int8x16_t product = vmulq_s8(input, weights);
    
    return vaddq_s8(acc, product);
}
```

---

## Component Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| `TriXLinear` | `kernel/bindings.py` | Base ternary linear |
| `TriXTile` | `nn/hierarchical.py` | Single specialist |
| `HierarchicalTriXFFN` | `nn/hierarchical.py` | Main FFN |
| `SparseLookupFFN` | `nn/sparse_lookup.py` | Routing-as-computation |
| `TemporalTileLayer` | `nn/temporal_tiles.py` | State-aware routing |
| `CompiledDispatch` | `nn/compiled_dispatch.py` | O(1) inference |

---

## Further Reading

- [Theory](THEORY.md) - Mathematical foundations
- [API Reference](API.md) - Complete API documentation
- [Benchmarks](BENCHMARKS.md) - Performance methodology
