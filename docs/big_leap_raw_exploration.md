# Big Leap: Raw Exploration

*Getting funky. Following threads. Seeing what wants to emerge.*

---

## Starting Point

We have signatures. We have tiles. We have routing.

Scale it up and it becomes memory.

But what KIND of memory? And what else is hiding here?

---

## Thread 1: The Hopfield Connection

Modern Hopfield networks store patterns as attractors. Retrieval is convergence to the nearest attractor.

TriX signatures are... attractors? Each signature defines a basin. Inputs fall into the basin they align with.

But Hopfield is recurrent (iterate until convergence). TriX is feedforward (one-shot argmax).

**What if we iterated?**

```python
# Current: One-shot routing
scores = input @ signatures.T
winner = scores.argmax()

# Alternative: Iterative refinement
x = input
for _ in range(steps):
    scores = x @ signatures.T
    soft_route = softmax(scores / temperature)
    x = soft_route @ tile_outputs  # Blend outputs
# Converge to attractor
```

Iterative routing as attractor dynamics. The input finds its home by settling.

**Wild thought:** What if the tiles ARE the attractors? The signature defines the attractor, the tile computation is what happens at the attractor.

---

## Thread 2: Attention IS Routing

Attention: Query matches keys, retrieves values.
TriX: Input matches signatures, routes to tiles.

They're the same pattern!

```
Attention:  softmax(Q @ K.T) @ V
TriX:       onehot(input @ signatures.T) @ tile_outputs
```

TriX is hard attention with learned "values" (the tile computations).

**What if we softened it?**

```python
# Soft routing (differentiable)
weights = softmax(input @ signatures.T / temperature)
output = sum(weights[t] * tile_forward(input, t) for t in tiles)
```

This is... sparse attention. Where keys are signatures and values are computed on-the-fly.

**Wild thought:** What if attention heads ARE tiles? Each head has a signature (its key projection defines what it responds to). Multi-head attention is parallel routing to specialized processors.

Can we unify attention and TriX routing?

---

## Thread 3: Signatures as Addresses

In traditional memory, addresses are arbitrary (0x0000, 0x0001, ...).
In content-addressable memory, addresses are the content itself.

TriX signatures are content-addresses. But they're not arbitrary - they emerge from what the tile learned.

**The tile's address IS its function.**

This is profound. The "where" and the "what" are unified. The signature tells you what the tile does because it's derived from how the tile works.

**What if we designed signatures deliberately?**

```python
# Instead of: signature emerges from weights
signature = tile.weight.sum(dim=0).sign()

# What about: signature is specified, weights are constrained
desired_signature = torch.tensor([1, -1, 1, 0, ...])
# Train tile such that signature converges to desired_signature
```

Controllable memory allocation. "I want a tile that responds to X" → design signature for X → train tile under that constraint.

---

## Thread 4: Hierarchical Signatures as a Tree

With hierarchy:
```
Root signature → Child signatures → ... → Leaf tiles
```

This is a tree. But it's a SEMANTIC tree. Each branch represents a region of input space.

**What does the tree structure tell us?**

If we cluster signatures, the clusters reveal how the model has organized its knowledge:
- Cluster 1: Numeric inputs
- Cluster 2: Punctuation contexts  
- Cluster 3: Common words
- ...

The hierarchy is an emergent ontology. The model has carved input space into categories, and the tree shows us how.

**Wild thought:** Can we visualize/inspect the tree to understand the model? The routing structure as explanation?

---

## Thread 5: Dynamic Signatures

Current signatures are fixed per forward pass (derived from current weights).

But what if signatures adapted to context?

```python
# Context-aware signatures
def dynamic_signature(tile, context):
    base_sig = tile.weight.sum(dim=0).sign()
    context_modulation = context_encoder(context)
    return base_sig * context_modulation.sign()
```

The same tile responds to different things based on context.

**Analogy:** A librarian who reorganizes based on who's asking. "Oh, you're interested in cooking? Let me highlight these sections..."

---

## Thread 6: Tiles as Programs

Each tile is a learned function. But with ternary weights, it's a very constrained function.

What can a ternary tile compute?
- Linear combination with coefficients {-1, 0, +1}
- Essentially: sum some inputs, subtract others, ignore the rest

This is like a voting function. Each output is a vote tally.

**Wild thought:** What if we thought of tiles as programs?
- Signature = Type signature (what inputs it accepts)
- Weights = Program (what it computes)
- Output = Return value

Routing is type-checking. The input's "type" (determined by signature alignment) determines which "function" runs.

This is... dependent types? Or something like it?

---

## Thread 7: Memory Consolidation

Biological memory consolidates. Short-term becomes long-term. Memories merge, compress, reorganize.

What would consolidation look like for TriX?

```python
def consolidate(model):
    # Find similar tiles
    sigs = model.get_all_signatures()
    similar_pairs = find_similar(sigs, threshold=0.9)
    
    # Merge similar tiles
    for (t1, t2) in similar_pairs:
        merged = merge_tiles(t1, t2)
        model.replace_tiles(t1, t2, merged)
    
    # Split overloaded tiles
    for tile in model.tiles:
        if tile.usage > threshold:
            new_tiles = split_tile(tile)
            model.replace_tile(tile, new_tiles)
```

The model reorganizes itself. Consolidates redundancy. Expands where needed.

**Wild thought:** A sleeping phase. Train during the day, consolidate at night. The model dreams (replays inputs, refines routing, merges memories).

---

## Thread 8: What's the Minimal Implementation?

Strip away everything non-essential. What's the core?

```python
class MemoryNetwork:
    def __init__(self, num_memories, dim):
        self.memories = [TernaryFunction(dim) for _ in range(num_memories)]
    
    def forward(self, x):
        # Content-addressed lookup
        signatures = [m.signature for m in self.memories]
        scores = x @ stack(signatures).T
        best = scores.argmax()
        
        # Compute at retrieved memory
        return self.memories[best](x)
```

That's it. That's the whole thing.

- `TernaryFunction`: Ternary weights, computes something
- `signature`: Sum and sign of weights
- Routing: Argmax over signature alignment

Everything else is optimization (hierarchy, training tricks, kernels).

---

## Thread 9: What Could Go Wrong?

**Collapse:** All inputs route to one tile. 
*Solution: Load balancing loss.*

**Interference:** New learning damages old memories.
*Solution: Protect signatures of old tiles? Freeze old tiles?*

**Capacity:** Not enough tiles for the task.
*Solution: Dynamic tile allocation. Grow as needed.*

**Hierarchy mistakes:** Wrong cluster → wrong tile.
*Solution: Soft hierarchy? Multiple paths? Error correction?*

**Signature drift:** Signatures change during training, routing becomes unstable.
*Solution: EMA signatures? Commit to signatures early?*

---

## Thread 10: The Biological Parallel

Cortical columns in the brain:
- Specialized processors
- Local computation
- Organized hierarchically
- Route signals based on... something

What if TriX tiles are like cortical columns?
- Each tile: a column with its specialty
- Signature: the column's "receptive field"
- Routing: signal propagation based on content

This is very speculative. But the structure rhymes.

---

## What's Emerging?

Several things want to converge:

1. **Signatures as addresses** - Content-addressing is the core primitive
2. **Hierarchy for scale** - Tree structure for O(log n) routing
3. **Attention connection** - TriX as hard attention with computed values
4. **Self-organization** - Consolidation, splitting, merging
5. **Interpretability** - The tree reveals learned structure

The engineering path: Start with hierarchy. The rest follows.

---

## The Funkiest Idea

What if we inverted everything?

Instead of: Train model → Extract signatures → Route
What about: Design signatures → Train tiles to match → Controlled memory

The signatures define the ontology. The training fills in the functions.

Like: "I want a tile for questions, a tile for statements, a tile for numbers..."

Design the routing first. Train the computation second.

**Prescribed hierarchy, learned content.**

---

## Raw Priority

If I had to build ONE thing next, it would be:

**Hierarchical routing with signature clustering.**

Because:
1. It's the scaling path
2. It reveals the memory structure
3. It's implementable now
4. Everything else builds on it

Two levels. Cluster signatures from tile signatures. Route coarse then fine.

That's the next step.

---

*End of raw exploration. Time to converge...*
