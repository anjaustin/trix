# Big Leap: Convergence

*Reflecting on the raw exploration. Finding the signal.*

---

## What Kept Appearing

Reading through my raw thoughts, certain ideas surfaced repeatedly:

1. **Hierarchy is essential** - Can't scale without it
2. **Signatures are addresses** - Content-addressing is the primitive
3. **Attention and routing are the same pattern** - Deep structural similarity
4. **Self-organization wants to happen** - Consolidation, splitting, growing
5. **Interpretability comes free** - The routing structure IS the explanation

Let me follow each to its conclusion.

---

## Convergence 1: Hierarchical Routing

**The insight:** O(n) routing doesn't scale. But signatures naturally cluster.

**The implementation:**
```python
class HierarchicalRouter:
    def __init__(self, tiles, num_clusters):
        # Cluster tiles by signature similarity
        tile_sigs = [t.signature for t in tiles]
        self.clusters = cluster(tile_sigs, num_clusters)
        
        # Cluster signature = mean of member signatures
        self.cluster_sigs = [
            mean([tile_sigs[i] for i in cluster]).sign()
            for cluster in self.clusters
        ]
    
    def route(self, x):
        # Coarse: find cluster
        cluster_scores = x @ stack(self.cluster_sigs).T
        cluster_idx = cluster_scores.argmax()
        
        # Fine: find tile within cluster
        tile_indices = self.clusters[cluster_idx]
        tile_sigs = [self.tile_sigs[i] for i in tile_indices]
        tile_scores = x @ stack(tile_sigs).T
        tile_idx = tile_indices[tile_scores.argmax()]
        
        return tile_idx
```

**Cost:** O(sqrt(n)) with 2 levels, O(log n) with more levels.

**What we gain:** Scale to 1000+ tiles without routing bottleneck.

---

## Convergence 2: Signatures as Content Addresses

**The insight:** The signature isn't arbitrary. It's derived from what the tile learned. Address = Function.

**The implication:** We can reason about what a tile does by inspecting its signature without running it.

```python
def what_does_tile_want(tile):
    sig = tile.signature  # [d_model]
    
    # Positive entries: wants these features
    wants = (sig == 1).nonzero()
    
    # Negative entries: wants opposite of these
    anti_wants = (sig == -1).nonzero()
    
    # Zero entries: doesn't care
    ignores = (sig == 0).nonzero()
    
    return wants, anti_wants, ignores
```

**What we gain:** Introspection. Debugging. Understanding.

---

## Convergence 3: Unifying Attention and Routing

**The insight:** Both are "match query to keys, retrieve values."

| | Attention | TriX Routing |
|-|-----------|--------------|
| Query | Q projection | Input |
| Keys | K projection | Signatures |
| Values | V projection | Tile outputs |
| Aggregation | Softmax weighted sum | Hard argmax |

**The implication:** TriX could use soft routing for training (differentiable), hard routing for inference (fast).

```python
def forward(self, x, hard=True):
    scores = x @ self.signatures.T
    
    if hard:
        # Inference: fast, sparse
        idx = scores.argmax()
        return self.tiles[idx](x)
    else:
        # Training: differentiable
        weights = softmax(scores / self.temperature)
        return sum(w * tile(x) for w, tile in zip(weights, self.tiles))
```

**What we gain:** Best of both worlds. Train soft, deploy hard.

---

## Convergence 4: Self-Organization

**The insight:** The model should manage its own memory structure.

**Key operations:**
1. **Consolidate:** Merge tiles with similar signatures (reduce redundancy)
2. **Split:** Divide overloaded tiles (increase capacity where needed)
3. **Prune:** Remove unused tiles (save resources)
4. **Grow:** Add tiles when capacity is exhausted (scale dynamically)

```python
class SelfOrganizingMemory:
    def consolidate(self):
        # Find and merge similar tiles
        similar = self.find_similar_tiles(threshold=0.95)
        for (t1, t2) in similar:
            self.merge(t1, t2)
    
    def split_overloaded(self):
        # Split tiles with too much traffic
        for tile in self.tiles:
            if tile.usage_rate > 0.5:  # Handles >50% of inputs
                self.split(tile)
    
    def prune_unused(self):
        # Remove tiles that never activate
        for tile in self.tiles:
            if tile.usage_rate < 0.001:
                self.remove(tile)
```

**What we gain:** Adaptive capacity. No manual architecture search.

---

## Convergence 5: Interpretability

**The insight:** The routing structure reveals learned organization.

If we cluster tiles and visualize the hierarchy:
```
Root
├── Cluster 0: "Numeric" (tiles that handle numbers)
│   ├── Tile 0: Arithmetic operations
│   ├── Tile 1: Digit sequences
│   └── Tile 2: Numeric comparisons
├── Cluster 1: "Linguistic" (tiles that handle words)
│   ├── Tile 3: Common words
│   ├── Tile 4: Rare words
│   └── Tile 5: Word boundaries
└── Cluster 2: "Structural" (tiles that handle syntax)
    ├── Tile 6: Punctuation
    ├── Tile 7: Whitespace
    └── Tile 8: Sentence boundaries
```

The model has learned an ontology. We can read it.

**What we gain:** Explanation. Trust. Debugging.

---

## The Engineering Path

Based on convergence, here's the build order:

### Phase 1: Hierarchical Routing (Foundation)
- Implement 2-level hierarchy
- Cluster signatures automatically
- Verify O(sqrt(n)) scaling

### Phase 2: Soft/Hard Training
- Soft routing during training (better gradients)
- Hard routing during inference (fast)
- Temperature annealing

### Phase 3: Self-Organization
- Tile usage tracking
- Consolidation (merge similar)
- Splitting (divide overloaded)

### Phase 4: Visualization & Introspection
- Signature inspector
- Routing tree visualization
- Per-tile analysis tools

---

## The Minimal Viable Big Leap

If I had to ship ONE thing:

**Hierarchical routing with automatic signature clustering.**

```python
class HierarchicalTriXFFN(nn.Module):
    def __init__(self, d_model, num_tiles, num_clusters):
        self.tiles = [SparseTriXTile(d_model) for _ in range(num_tiles)]
        self.num_clusters = num_clusters
        self.cluster_assignments = None
        self.cluster_signatures = None
        
    def build_hierarchy(self):
        """Cluster tiles by signature similarity."""
        sigs = torch.stack([t.get_signature() for t in self.tiles])
        
        # K-means on signatures
        centroids, assignments = kmeans(sigs, self.num_clusters)
        
        self.cluster_signatures = centroids.sign()
        self.cluster_assignments = assignments
    
    def route(self, x):
        """Two-level routing: cluster then tile."""
        # Level 1: Find cluster
        cluster_scores = x @ self.cluster_signatures.T
        cluster = cluster_scores.argmax(dim=-1)
        
        # Level 2: Find tile within cluster
        tile_mask = (self.cluster_assignments == cluster.unsqueeze(-1))
        # ... route within cluster
        
        return tile_idx
```

This alone enables scaling to hundreds of tiles.

---

## What's Deferred

Not for the first implementation:
- Soft routing (keep hard for simplicity)
- Self-organization (manual for now)
- Visualization (later)
- Dynamic growth (fixed tile count first)

Keep it simple. Prove hierarchy works. Then add complexity.

---

## Success Criteria

The Big Leap is validated if:

1. **64+ tiles** work without routing bottleneck
2. **Quality maintained** at 2% of dense baseline
3. **Routing cost** is O(sqrt(n)) or better
4. **Tiles specialize** more with more tiles
5. **Hierarchy is interpretable** (clusters make sense)

---

*Ready to synthesize into an engineering plan...*
