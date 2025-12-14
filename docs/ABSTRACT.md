# TriX: True 2-Bit Sparse Ternary Networks with Emergent Routing

## Abstract

We introduce TriX, a neural network architecture that combines true 2-bit weight packing with tile-based conditional computation and zero-parameter emergent routing. Unlike approaches that constrain weights to ternary values {-1, 0, +1} while storing them in higher-precision formats, TriX physically packs four weights per byte and operates directly on this compressed representation via optimized ARM NEON kernels, achieving 16x memory compression over float32.

TriX organizes computation into tiles and routes each input to a single tile based on *weight signatures*—ternary vectors derived by summing and signing each tile's weights. This emergent routing requires no learned parameters: the weights themselves encode routing preferences. Inputs align with the tile whose signature best matches their pattern, enabling conditional computation that skips 75% of operations per forward pass.

We demonstrate that naive application of sparse inference to densely-trained models fails catastrophically (perplexity degradation from 5.04 to 81.35). However, training with sparse routing from initialization—where each tile learns independently—achieves near-dense quality (perplexity 5.14) while maintaining 4x computational sparsity. Tiles naturally specialize by input type without explicit supervision.

Our key contributions are:

1. **True 2-bit implementation**: Physical packing and kernels that operate on compressed weights, not just ternary constraints
2. **Emergent routing**: Zero-parameter routing derived from weight structure, replacing learned gate networks
3. **Sparse training validation**: Demonstrating that tiles can learn complete, independent representations when trained with routing from the start
4. **The "read, don't learn" principle**: Showing that routing information is already encoded in ternary weights and can be extracted rather than learned

TriX achieves 4.26x measured speedup at 75% sparsity with only 2% quality degradation versus dense baselines. The approach suggests a broader principle: neural network structure encodes more information than typically extracted, and reading this structure can replace learned components.

---

## Key Results

| Configuration | Perplexity | Compute |
|---------------|------------|---------|
| Dense baseline | 5.04 | 100% |
| Dense-trained, sparse-inference | 81.35 | 25% |
| **Sparse-trained, sparse-inference** | **5.14** | **25%** |

## The Core Insight

```python
# Emergent routing in three lines
signature = tile_weights.sum(dim=0).sign()  # What the tile wants
score = input @ signature                    # How well input matches
route = (score == scores.max())              # Send to best match
```

Ternary weights are votes. Signatures encode preferences. Routing emerges from alignment.

---

*Don't learn what you can read.*
