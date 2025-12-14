# Zero Routing Reflections

*Examining the raw thoughts for nodes of opportunity*

---

## Reflection 1: The "No Gradient Needed" Property

From raw thoughts:
> "the routing decision doesn't need gradients. Weights get gradients from the main forward pass. Routing adapts as a side effect."

This is profound. We've been fighting the gradient problem for routing, but emergent routing sidesteps it entirely.

**Opportunity:** This could be a general principle. What other "decisions" in neural networks are we trying to learn directly that could instead emerge from existing structure?

- Attention patterns? (emerge from key-query alignment - already done!)
- Layer skipping? (emerge from layer weight norms?)
- Channel pruning? (emerge from weight magnitude?)

The pattern: **don't learn the decision, read it from the weights.**

---

## Reflection 2: Signatures as a First-Class Concept

The signature isn't just a routing trick - it's a **semantic summary** of what a tile cares about.

```python
signature = weight.sum(dim=0).sign()
```

This is the tile's "preference vector" - a 1D ternary representation of its entire weight matrix.

**Opportunity:** Signatures could be useful beyond routing:
- **Tile visualization:** Plot signatures to understand specialization
- **Tile similarity:** Cosine similarity between signatures shows redundancy
- **Tile pruning:** Remove tiles with similar signatures
- **Transfer learning:** Match signatures between models for tile-to-tile transfer
- **Debugging:** "Why did this input route here?" - show input-signature alignment

Could add a `TriXAnalyzer` class for signature-based model introspection.

---

## Reflection 3: The Caching Insight

From raw thoughts:
> "extend pack() to also precompute and cache signatures"

This is clean. `pack()` already exists for converting weights to 2-bit format. Extending it to cache signatures keeps the API simple.

**Opportunity:** The pack/unpack pattern could become a more general "mode switching":
- `train()` mode: full precision weights, dynamic signatures
- `eval()` + `pack()` mode: 2-bit weights, cached signatures, NEON kernel

Could rename to something clearer:
- `layer.compile()` - prepare for fast inference
- `layer.decompile()` - return to training mode

Or keep `pack()`/`unpack()` for simplicity.

---

## Reflection 4: Top-K Extension

From raw thoughts:
> "For higher tile counts (16, 64), might want top-2 or top-4"

Top-k routing is natural with signature scoring. But there's a deeper opportunity here.

**Opportunity:** Adaptive k based on score distribution.

If one tile has a much higher score than others, use just that tile.
If scores are close, use multiple tiles.

```python
scores = input @ signatures.T
max_score = scores.max(dim=-1)
second_score = scores.topk(2, dim=-1).values[:, 1]
gap = max_score - second_score

# Confident -> 1 tile, uncertain -> 2 tiles
k = torch.where(gap > threshold, 1, 2)
```

This is **confidence-based routing**. Route to more tiles when uncertain. Could be a nice follow-up feature.

---

## Reflection 5: The Naming Question

From raw thoughts:
> "I like 'Zero Routing' for marketing, 'Signature-based routing' for technical docs"

Names matter. They shape how people think about the concept.

- "Zero Routing" emphasizes what we removed (learned gates)
- "Emergent Routing" emphasizes how it arises (from structure)
- "Signature Routing" emphasizes the mechanism
- "Self-Routing" emphasizes autonomy

**Opportunity:** Use different names for different contexts:
- Paper/marketing: "Zero-Parameter Routing" (sounds impressive)
- API: `TriXFFN` (simple, implies "the TriX way")
- Technical docs: "Signature-based emergent routing"
- Code comments: "routing from weight signatures"

---

## Reflection 6: What If Signatures Collapse?

From raw thoughts:
> "What if all tiles converge to similar signatures? ... if task needs diversity, tiles will differentiate to reduce loss"

This is optimistic but might not always hold. 

**Opportunity:** Monitor signature diversity during training. If it drops below a threshold, intervene.

Options:
1. **Soft:** Log a warning, let user decide
2. **Medium:** Add optional diversity regularization
3. **Hard:** Reinitialize collapsed tiles

For now, monitoring is enough. Add `get_signature_diversity()` method:
```python
def get_signature_diversity(self):
    sigs = self.get_tile_signatures()
    # Pairwise differences
    diversity = 0
    for i in range(self.num_tiles):
        for j in range(i+1, self.num_tiles):
            diversity += (sigs[i] != sigs[j]).float().mean()
    return diversity / (self.num_tiles * (self.num_tiles - 1) / 2)
```

---

## Reflection 7: Integration Depth

How deep should zero routing go?

**Shallow:** Just add TriXFFN, keep everything else
**Medium:** Update TriXLinear to support auto-routing
**Deep:** Make zero routing the default everywhere, deprecate learned gates

**Opportunity:** Start shallow, go deeper based on feedback.

Phase 1: Add TriXFFN as new option (non-breaking)
Phase 2: Add auto-routing to TriXLinear (non-breaking, opt-in)
Phase 3: Make TriXFFN the recommended default (docs change)
Phase 4: Deprecate GatedFFN (breaking, major version)

This gives users time to migrate and provides escape hatches.

---

## Reflection 8: The Bigger Picture

Emergent routing fits into a larger philosophy:

> **TriX Principles:**
> 1. Discrete is not a bug, it's a feature (ternary weights)
> 2. Structure encodes information (signatures)
> 3. Emergence over engineering (routing from weights)
> 4. Simplicity wins (3 lines vs gate network)

**Opportunity:** Document these principles. Make them part of the TriX identity. Future features should align with them.

This could guide future development:
- What else can emerge from ternary structure?
- What other "learned" components can be replaced with "read" components?
- Where else does simplicity beat complexity?

---

## Reflection 9: Testing Philosophy

From raw thoughts on testing:
> "consistency, discrimination, gradient flow, inference speedup, API compatibility"

These are good functional tests. But we should also test the **emergent properties**.

**Opportunity:** Property-based tests:
- "Routing should be deterministic given same weights"
- "Small input changes should usually not change routing"
- "Large input changes should sometimes change routing"
- "Training should eventually stabilize routing"

These test the emergent behavior, not just the implementation.

---

## Reflection 10: Documentation as Design

Writing these docs has clarified the design. The act of explaining forces precision.

**Opportunity:** Write the user guide BEFORE finalizing implementation. 

What would a tutorial look like?
```python
# The TriX Way: Emergent Routing
from trix import TriXFFN

# Create an FFN - routing happens automatically
ffn = TriXFFN(d_model=512, num_tiles=4)

# Forward pass - no gate needed!
output = ffn(input)

# Inspect routing
ffn.get_routing_stats(input)  # See where inputs went
ffn.get_tile_signatures()     # See what each tile wants
```

If this feels right, the design is right.

---

## Key Opportunities Summary

1. **Signatures as first-class citizens** - visualization, similarity, pruning, transfer
2. **Confidence-based adaptive k** - route to more tiles when uncertain
3. **Diversity monitoring** - detect and warn about signature collapse
4. **Phased rollout** - shallow to deep integration
5. **Principle documentation** - codify the TriX philosophy
6. **Property-based testing** - test emergence, not just mechanics
7. **Doc-driven design** - write tutorials first

---

## Next: Synthesis

Take these reflections and converge on a concrete implementation plan.
