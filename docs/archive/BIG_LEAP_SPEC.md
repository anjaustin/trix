# The Big Leap: TriX as Content-Addressable Memory

*A spec for lab partners - seeking fresh perspectives*

---

## Context

We've built TriX, a neural network architecture with:

1. **True 2-bit weights** - Ternary {-1, 0, +1}, physically packed (4 weights/byte)
2. **Tile-based computation** - Model divided into independent tiles
3. **Emergent routing** - Zero-parameter routing via weight signatures
4. **Sparse training** - Each tile learns independently, achieves 2% of dense quality

**Current results:** 4 tiles, 4x speedup, PPL 5.14 vs 5.04 dense baseline.

---

## The Insight

As we scale tiles, something emerges:

| Tiles | Routing Complexity | What It Feels Like |
|-------|-------------------|-------------------|
| 4 | O(4) - trivial | Sparse MoE |
| 64 | O(64) - manageable | Needs optimization |
| 1000+ | O(1000) - expensive | Needs hierarchy |

But with hierarchical signatures, routing becomes O(log n):

```
Input → Match coarse signature → Match fine signature → Tile
```

And at that point, we realize: **this is content-addressable memory**.

- Signatures = Keys
- Tiles = Values (learned functions)
- Routing = Lookup by content alignment
- Forward pass = Retrieve matching memory, compute

---

## The Core Mechanism

### Current (4 tiles)
```python
signatures = [tile.weight.sum(dim=0).sign() for tile in tiles]  # [4, d_model]
scores = input @ signatures.T  # [batch, 4]
winner = scores.argmax(dim=-1)  # Route to best match
```

### Proposed (Hierarchical)
```python
# Level 1: Coarse routing (e.g., 32 clusters)
cluster_sigs = get_cluster_signatures()  # [32, d_model]
cluster_scores = input @ cluster_sigs.T
cluster = cluster_scores.argmax(dim=-1)

# Level 2: Fine routing (e.g., 32 tiles per cluster = 1024 total)
tile_sigs = get_tile_signatures(cluster)  # [32, d_model]
tile_scores = input @ tile_sigs.T
tile = tile_scores.argmax(dim=-1)

# Total: 1024 tiles, but only 64 comparisons (32 + 32)
```

---

## What This Enables

### 1. Massive Specialization
With 1000+ tiles, each can specialize narrowly:
- This tile handles "questions about locations"
- That tile handles "arithmetic with small numbers"
- Another handles "formal greetings"

### 2. Interpretable Routing
We can inspect which "memories" activate for which inputs. The routing pattern IS the explanation.

### 3. Incremental Learning
Add new tiles without retraining everything. Just add a new memory slot with its signature.

### 4. Compositional Retrieval
Combine signatures algebraically:
```python
# "What handles questions AND locations?"
combined_sig = question_sig + location_sig
```

### 5. Graceful Scaling
Memory grows by adding tiles. Routing stays O(log n). No architectural changes needed.

---

## Open Questions for Vi and VGem

### Architecture Questions
1. How deep should the hierarchy be? 2 levels? 3? Adaptive?
2. How do we learn/update cluster signatures? Fixed from tile signatures? Learned separately?
3. Should clusters overlap? (Soft hierarchy vs hard hierarchy)

### Training Questions
4. Train flat then cluster? Or hierarchical from the start?
5. How do we prevent cluster collapse? (All inputs → one cluster)
6. Load balancing at cluster level vs tile level?

### Memory Questions
7. Is there a maximum useful number of tiles? When does more not help?
8. Can tiles be dynamically allocated? (Grow memory as needed)
9. How do we handle tile "forgetting" or "interference"?

### Theoretical Questions
10. What's the relationship to Hopfield networks / modern Hopfield networks?
11. Is there a connection to attention mechanisms? (Attention as soft routing?)
12. How does this relate to memory-augmented neural networks (NTMs, DNCs)?

---

## The Hypothesis

**TriX at scale is not a neural network with routing. It's a memory system with learned content.**

The tiles aren't "experts" in the MoE sense. They're **memory slots** containing learned functions, indexed by ternary signatures.

Inference isn't "forward pass." It's **content-addressed retrieval** followed by computation.

---

## What We're Asking

We've followed emergent routing from 4 tiles to its logical conclusion. We think it becomes content-addressable memory.

**Are we seeing this right?**

What are we missing? What related work should we know about? What pitfalls do you see?

Fresh eyes welcome.

---

## Current Codebase

```
trix/
├── src/trix/
│   ├── kernel/       # 2-bit NEON kernel
│   │   ├── trix.cpp  # Packed ternary matmul
│   │   └── bindings.py  # Signatures, pack/unpack
│   ├── nn/
│   │   ├── trix.py   # TriXFFN (4-tile emergent routing)
│   │   └── sparse.py # SparseTriXFFN (sparse training)
│   └── qat/          # Quantization-aware training
├── tests/            # 90 tests passing
└── docs/             # Journey documentation
```

**Entry point:** `SparseTriXFFN` in `src/trix/nn/sparse.py`

---

*We're listening. What do you see?*
