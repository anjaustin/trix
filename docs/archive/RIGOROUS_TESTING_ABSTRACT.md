# Rigorous Testing of Hierarchical TriX: Findings and Implications

## Abstract

We present a systematic empirical analysis of HierarchicalTriXFFN, a 2-level content-addressable memory architecture where routing is derived from ternary weight signatures rather than learned parameters. Through eight rigorous tests covering signature quality, routing consistency, tile specialization, cluster coherence, gradient flow, load distribution, scaling behavior, and training convergence, we establish the following findings:

**Signatures are discriminative.** At 64 tiles, 100% of signatures are unique with an average Hamming distance of 0.66, indicating no collision risk. Signatures remain stable under input perturbations up to 20% noise magnitude.

**Emergent routing produces specialization without training.** Different input patterns route to different tiles based purely on signature alignment—before any gradient updates occur. In controlled experiments, 6 distinct synthetic patterns routed to 6 unique primary tiles with concentration rates up to 100%. This confirms the core hypothesis: routing information is encoded in weight structure and can be read rather than learned.

**Balanced clustering is essential.** Standard k-means on random initialization signatures produces severe imbalance (cluster sizes ranging 3-23 for 64 tiles). We introduce balanced k-means with size constraints, achieving uniform cluster sizes and reducing tile usage imbalance from 7.3x to 1.9x (max/min ratio).

**Random signatures lack natural cluster structure.** The Calinski-Harabasz index of 1.31 and only 3.6% improvement over random assignment indicate that initial ternary signatures are approximately uniformly distributed on the hypersphere. Clustering provides routing efficiency (O(sqrt(n))), not semantic organization. Meaningful clusters may emerge through training as signatures evolve.

**Gradients flow correctly.** All activated tiles receive gradients, with gradient norms proportional to activation frequency. The sparse routing mechanism preserves differentiability through straight-through estimation.

**The architecture scales efficiently.** Parameter count grows only 1.1x from 4 to 64 tiles (due to inverse scaling of per-tile hidden dimension), while forward pass time grows 3.5x—sublinear in tile count due to hierarchical routing.

**Key Limitation:** In our training convergence test, task accuracy remained at chance level (25%) despite 50 epochs, though routing concentration improved from 6.2% to 54.5%. This suggests the sparse ternary computation may require architectural modifications (residual connections, normalization) or training adjustments (learning rate, batch size) for effective task learning.

## Implications

1. **Zero-parameter routing is validated.** The signature-based routing mechanism works as theorized—inputs find appropriate tiles through alignment, not learned gating.

2. **Content-addressable memory interpretation is supported.** The system behaves as a key-value store where signatures are keys and tile computations are values. Routing is retrieval by content similarity.

3. **Balanced clustering is a requirement, not an optimization.** Without size constraints, k-means produces degenerate clusterings that undermine the hierarchical routing benefit.

4. **Training dynamics need investigation.** The disconnect between routing specialization and task learning suggests the computation within tiles may need strengthening, or the routing-to-learning feedback loop needs refinement.

## Validated Properties

| Property | Test Result | Confidence |
|----------|-------------|------------|
| Signature uniqueness | 100% at 64 tiles | High |
| Routing determinism | Identical across runs | High |
| Routing stability | Stable to 20% noise | High |
| Tile utilization | 16/16 tiles active | High |
| Load balance (with fix) | 1.9x max/min ratio | High |
| Pattern specialization | 6/6 patterns discriminated | High |
| Gradient correctness | All active tiles receive gradients | High |
| Cluster coherence | Low (1.31 CH index) | Expected |
| Task convergence | Not achieved in test | Needs work |

## Open Questions for Vi and VGem

1. **Why does routing specialize but task learning stall?** The tiles route correctly but don't produce useful features. Is this a capacity issue, gradient flow issue, or fundamental limitation?

2. **Should we abandon clustering for flat routing until signatures are trained?** Given low cluster coherence at init, is hierarchical routing premature before training shapes signatures?

3. **Is the signature-from-weights approach optimal?** VGem suggested random projections as an alternative to column-sum signatures. Should we explore this given the uniformity of random init signatures?

4. **What's the minimum viable architecture for task learning?** Do we need residual connections, layer normalization, or multiple hierarchical blocks to achieve convergence?

5. **How do we validate the "Qdrant with a brain" hypothesis at scale?** What experiments would definitively show this is content-addressable memory with executable functions?

---

*We have verified the routing mechanism. The question now is whether the computation at each address is sufficient for learning.*
