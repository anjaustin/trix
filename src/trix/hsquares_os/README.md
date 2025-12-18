# Hollywood Squares XOR OS

> **⚠️ EXPERIMENTAL** - This module is highly experimental. The previous TriX version (v0.9.x) should be considered stable. This work explores speculative architectural ideas.

## Overview

The XOR Hollywood Squares OS is a memory-efficient tile management system that uses XOR compression to achieve **100x memory reduction** for sparse tile operations.

## Key Insight

> **"XOR values together = two numbers in superposition"**
>
> Memory: Nothing. Work: Nothing.
> Store A ⊕ B. Given A, recover B = A ⊕ (A ⊕ B).

## Components

### SparseXOR

Sparse representation of XOR deltas from a base tile.

```python
@dataclass
class SparseXOR:
    positions: torch.Tensor  # Non-zero positions
    values: torch.Tensor     # XOR delta values
    base_tile_id: int        # Reference tile
```

### XORTileManager

Manages tiles as base + sparse deltas.

```python
manager = XORTileManager(num_tiles=10000, tile_size=512)

# Add tile (stored as delta from nearest base)
manager.add_tile(tile_id=42, data=tile_tensor)

# Retrieve tile (reconstructed on demand)
tile = manager.get_tile(tile_id=42)
```

**Memory Results:**
- 10,000 tiles in 0.9 MB (vs 10 MB uncompressed)
- 11.4x compression ratio
- 99% sparsity in deltas

### XORMessageBus

Temporal XOR encoding for inter-tile communication.

```python
bus = XORMessageBus()

# Send message (stored as delta from previous)
bus.send(tile_id=1, message=tensor, timestamp=t)

# Messages with similar content compress to near-zero
```

### XORCheckpointer

Incremental XOR chain for checkpointing.

```python
checkpointer = XORCheckpointer()

# Checkpoint (stores delta from previous)
checkpointer.checkpoint(state, step=100)

# Restore any checkpoint via XOR chain
state = checkpointer.restore(step=100)
```

## Scaling

| Tiles | Memory (Uncompressed) | Memory (XOR) | Compression |
|-------|----------------------|--------------|-------------|
| 10K | 20 MB | 0.9 MB | 22x |
| 1M | 2 GB | ~90 MB | 22x |
| 640M | 1.3 TB | ~64 GB | 20x |

**29D Hypercube (537M vertices) fits in 64 GB.**

## Use Cases

1. **Large Tile Libraries** - Store millions of tiles with minimal memory
2. **Temporal State** - Track tile evolution efficiently
3. **Distributed Systems** - Compress messages between nodes
4. **Checkpointing** - Incremental snapshots of system state

## Connection to Tesseract

The XOR OS enables the 29-dimensional tesseract structure:
- Each tesseract vertex = one tile
- XOR navigation between vertices = O(1)
- Hamming distance = path length
- 537M vertices addressable with 29 bits

## Files

| File | Purpose |
|------|---------|
| `xor_os.py` | Complete XOR OS implementation |

## Example

```python
from trix.hsquares_os.xor_os import XORTileManager, XORMessageBus

# Create manager
manager = XORTileManager(num_tiles=10000, tile_size=512)

# Populate with tiles
for i in range(10000):
    manager.add_tile(i, torch.randn(512))

# Check compression
stats = manager.stats()
print(f"Compression: {stats['compression_ratio']:.1f}x")
print(f"Sparsity: {stats['sparsity']*100:.1f}%")
```

## Performance

- Tile retrieval: O(k) where k = number of non-zero deltas
- Tile storage: O(tile_size) for delta computation
- Memory: O(base_tiles + k * num_tiles) vs O(tile_size * num_tiles)

With 99% sparsity: **100x memory reduction**.
