# Big Leap: Engineering Plan (Draft)

*Synthesizing exploration and convergence into an implementation plan.*

---

## The Goal

Scale TriX from 4 tiles to 64+ tiles with hierarchical routing, maintaining quality and achieving O(sqrt(n)) routing cost.

---

## Core Insight

At scale, TriX is content-addressable memory:
- **Signatures = Keys** (what each tile responds to)
- **Tiles = Values** (learned functions)
- **Routing = Lookup** (find best matching key)
- **Hierarchy = Indexing** (organize keys for fast lookup)

---

## Architecture

### HierarchicalTriXFFN

```
Input (batch, seq, d_model)
    │
    ▼
┌───────────────────────────────┐
│  Cluster Routing (Level 1)    │
│  scores = x @ cluster_sigs.T  │
│  cluster = argmax(scores)     │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│  Tile Routing (Level 2)       │
│  scores = x @ tile_sigs[c].T  │
│  tile = argmax(scores)        │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│  Tile Computation             │
│  output = tiles[tile](x)      │
└───────────────────────────────┘
    │
    ▼
Output (batch, seq, d_model)
```

### Key Components

1. **Tiles:** Ternary linear layers with signatures (existing `SparseTriXFFN` tiles)
2. **Clusters:** Groups of tiles with similar signatures
3. **Cluster Signatures:** Representative signature per cluster (centroid, signed)
4. **Two-Level Router:** Coarse (cluster) then fine (tile within cluster)

---

## Implementation Plan

### Step 1: HierarchicalTriXFFN Class

```python
class HierarchicalTriXFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_tiles = num_tiles
        self.num_clusters = num_tiles // tiles_per_cluster
        self.tiles_per_cluster = tiles_per_cluster
        
        # Create all tiles
        self.tiles = nn.ModuleList([
            TriXTile(d_model, d_hidden) 
            for _ in range(num_tiles)
        ])
        
        # Clustering state (built after init or during training)
        self.register_buffer('cluster_signatures', None)
        self.register_buffer('cluster_assignments', None)
        
    def build_hierarchy(self):
        """Cluster tiles by signature similarity using k-means."""
        with torch.no_grad():
            # Get all tile signatures
            sigs = torch.stack([
                tile.get_signature() for tile in self.tiles
            ])  # [num_tiles, d_model]
            
            # K-means clustering
            cluster_assignments, centroids = self._kmeans(
                sigs, self.num_clusters
            )
            
            # Store cluster signatures (signed centroids)
            self.cluster_signatures = centroids.sign()
            self.cluster_assignments = cluster_assignments
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, dict]:
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            output: [batch, seq, d_model]
            routing: [batch, seq] tile indices
            aux: auxiliary info dict
        """
        batch, seq, d = x.shape
        x_flat = x.view(-1, d)  # [batch*seq, d_model]
        
        # Level 1: Route to cluster
        cluster_scores = x_flat @ self.cluster_signatures.T  # [n, num_clusters]
        cluster_idx = cluster_scores.argmax(dim=-1)  # [n]
        
        # Level 2: Route to tile within cluster
        tile_idx = self._route_within_clusters(x_flat, cluster_idx)  # [n]
        
        # Compute outputs per tile
        output = self._compute_sparse(x_flat, tile_idx)  # [n, d_model]
        
        return (
            output.view(batch, seq, d),
            tile_idx.view(batch, seq),
            {'cluster_idx': cluster_idx.view(batch, seq)}
        )
    
    def _route_within_clusters(self, x: Tensor, cluster_idx: Tensor) -> Tensor:
        """Route each input to best tile within its assigned cluster."""
        tile_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        for c in range(self.num_clusters):
            mask = (cluster_idx == c)
            if not mask.any():
                continue
                
            # Get tiles in this cluster
            tile_indices = (self.cluster_assignments == c).nonzero().squeeze(-1)
            tile_sigs = torch.stack([
                self.tiles[i].get_signature() for i in tile_indices
            ])
            
            # Route within cluster
            scores = x[mask] @ tile_sigs.T
            local_winner = scores.argmax(dim=-1)
            tile_idx[mask] = tile_indices[local_winner]
        
        return tile_idx
```

### Step 2: Training with Hierarchy

```python
class HierarchicalTrainer:
    def __init__(self, model, rebuild_every=100):
        self.model = model
        self.rebuild_every = rebuild_every
        self.step = 0
    
    def train_step(self, x, y):
        # Periodically rebuild hierarchy as signatures evolve
        if self.step % self.rebuild_every == 0:
            self.model.build_hierarchy()
        
        # Forward with current hierarchy
        output, routing, aux = self.model(x)
        
        # Standard loss
        loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))
        
        # Balance loss (per cluster AND per tile)
        cluster_balance = self._cluster_balance_loss(aux['cluster_idx'])
        tile_balance = self._tile_balance_loss(routing)
        
        total_loss = loss + 0.01 * cluster_balance + 0.01 * tile_balance
        
        self.step += 1
        return total_loss
```

### Step 3: Efficient Routing (Optimized)

For production, avoid Python loops:

```python
def _route_within_clusters_fast(self, x: Tensor, cluster_idx: Tensor) -> Tensor:
    """Vectorized within-cluster routing."""
    # Precompute all tile signatures grouped by cluster
    # self.clustered_sigs: [num_clusters, tiles_per_cluster, d_model]
    
    # Gather signatures for each input's cluster
    cluster_sigs = self.clustered_sigs[cluster_idx]  # [n, tiles_per_cluster, d]
    
    # Batch dot product
    scores = torch.einsum('nd,ntd->nt', x, cluster_sigs)  # [n, tiles_per_cluster]
    
    # Local winner
    local_winner = scores.argmax(dim=-1)  # [n]
    
    # Convert to global tile index
    tile_idx = cluster_idx * self.tiles_per_cluster + local_winner
    
    return tile_idx
```

---

## Testing Plan

### Unit Tests

1. `test_hierarchy_construction` - Clustering produces valid assignments
2. `test_two_level_routing` - Inputs route correctly through both levels
3. `test_routing_cost` - Verify O(sqrt(n)) comparisons
4. `test_gradient_flow` - Gradients reach all tiles
5. `test_cluster_balance` - No cluster collapse
6. `test_tile_balance` - Tiles within clusters are used
7. `test_signature_stability` - Signatures don't drift wildly

### Integration Tests

8. `test_64_tiles_training` - Train converges with 64 tiles
9. `test_quality_vs_4_tiles` - Quality comparable to 4-tile version
10. `test_scaling_to_256` - Works at larger scale

### Validation

11. `test_ppl_on_shakespeare` - PPL within 5% of dense baseline
12. `test_routing_interpretability` - Clusters have semantic meaning

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Tiles | 64+ (16x current) |
| Routing cost | O(sqrt(n)) |
| Quality (PPL) | Within 5% of dense |
| Cluster balance | Max cluster usage < 2x mean |
| Tile balance | Max tile usage < 3x mean |

---

## Open Questions (For Vi/VGem Feedback)

1. **Fixed vs adaptive hierarchy?** Rebuild every N steps, or continuously adapt?
2. **Cluster count?** sqrt(num_tiles)? Or learned?
3. **Soft routing for training?** Or hard all the way?
4. **Balance loss weight?** 0.01 seems standard, but may need tuning.
5. **Signature initialization?** Random? Orthogonal? Semantic?

---

## Timeline Estimate

| Phase | Effort |
|-------|--------|
| HierarchicalTriXFFN core | 2-3 hours |
| Training loop adaptation | 1-2 hours |
| Tests (12 tests) | 2-3 hours |
| Validation on Shakespeare | 1-2 hours |
| **Total** | **6-10 hours** |

---

## What's Deferred

- Soft routing (keep hard for v1)
- Self-organization (manual clustering for v1)
- Visualization tools (later)
- Dynamic tile allocation (fixed count for v1)
- 3+ level hierarchy (2 levels sufficient to 256 tiles)

---

## The Deliverable

A working `HierarchicalTriXFFN` that:
1. Supports 64+ tiles
2. Routes in O(sqrt(n))
3. Trains end-to-end
4. Maintains TriX quality guarantees
5. Produces interpretable routing structure

---

*Awaiting feedback from Vi and VGem before implementation.*
