# Nodes of Opportunity

*Extrapolate. Get practical. Get radical. Find the edges.*

---

## Node 1: The Decoupled Tile

**The Idea**: Separate the "address" (what inputs route here) from the "function" (what happens to them).

**Practical Version**:
```python
class DecoupledTile:
    signature: nn.Parameter  # Learned address, NOT derived from weights
    compute: SplineModule    # Function, independent of routing
```

The signature would be learned end-to-end but wouldn't constrain the computation.

**Why This Might Work**:
- Current TriX couples signature to computation via `signature = weights.sum().sign()`
- This coupling might force suboptimal trade-offs
- Decoupling lets each optimize independently

**Why This Might Fail**:
- Loses the "zero routing parameters" elegance
- Might lead to signature collapse (all tiles want same inputs)
- Need explicit load balancing

**Radical Extension**:
What if signatures were DISCOVERED, not learned? Clustering algorithm on input space → signatures emerge from data distribution.

**Verdict**: Medium risk, medium reward. Worth a quick experiment.

---

## Node 2: Fractal Routing (Routing All The Way Down)

**The Idea**: Apply emergent routing at every level - cluster, tile, AND spline cell.

**Practical Version**:
```python
def forward(x):
    cluster_idx = route_to_cluster(x)  # Level 1
    tile_idx = route_to_tile(x, cluster_idx)  # Level 2
    cell_idx = route_to_cell(x, tile_idx)  # Level 3 (NEW)
    return blend_cell_outputs(x, cell_idx)
```

Level 3 routing replaces `idx = floor(x / stride)` with `idx = argmax(x @ cell_signatures)`.

**Why This Might Work**:
- Self-similar architecture (elegant, might scale better)
- Cell signatures could capture more nuanced distinctions
- Aligns with the "routing IS intelligence" philosophy

**Why This Might Fail**:
- More routing overhead (three levels of comparisons)
- Cell-level signatures might be too fine-grained to be meaningful
- Diminishing returns?

**Radical Extension**:
Make the number of routing levels adaptive. For "easy" inputs, route coarsely. For "hard" inputs, add more levels.

**Verdict**: High risk, potentially high reward. The elegant version of PQH.

---

## Node 3: Ternary Splines

**The Idea**: Quantize spline coefficients to {-1, 0, +1}.

**Practical Version**:
```python
class TernarySpline:
    # Per-cell: base ∈ {-1,0,+1}, slope ∈ {-1,0,+1}
    base_codes: int8   # 2 bits actual
    slope_codes: int8  # 2 bits actual
    scale: float       # Per-tile or per-layer

    def forward(x):
        cell = get_cell(x)
        base = self.base_codes[cell] * self.scale
        slope = self.slope_codes[cell] * self.scale
        return base + slope * x
```

**Why This Might Work**:
- Maximum compression: 4 bits per cell (vs. 64 bits for two floats)
- Still piecewise linear (universal approximator)
- Constraint might act as regularizer

**Why This Might Fail**:
- Extremely coarse: only 9 possible (base, slope) pairs
- Might need many more cells to compensate
- Training instability with STE?

**Radical Extension**:
What about 5-level quantization? {-2, -1, 0, +1, +2}. That's 3 bits per coefficient. 25 combinations per cell.

Or: different quantization levels for different layers. Early layers coarse, late layers fine.

**Verdict**: High risk, could be transformative. The "true 2-bit" spline.

---

## Node 4: The Compression Funnel

**The Idea**: Aggressively compress input to 2D, then use Spline2D lookup.

**Practical Version**:
```python
class CompressionFunnel:
    def forward(x):  # x: [batch, d_model]
        h = self.bottleneck(x)  # [batch, 2]
        h_q = quantize_to_grid(h, grid_size=16)  # [batch, 2] integers
        y_q = self.spline_2d[h_q[0], h_q[1]]  # [batch, d_hidden]
        return self.expand(y_q)  # [batch, d_model]
```

**Why This Might Work**:
- IF routing constrains input to a small region, 2D might capture it
- 16x16 = 256 states is actually a lot for a constrained space
- This is Nova's Path 3 made concrete

**Why This Might Fail**:
- 512D → 2D is brutal compression
- Bottleneck might discard critical information
- The "IF routing constrains" assumption might be wrong

**Radical Extension**:
Adaptive dimensionality. Start with 2D during inference. If output confidence is low, fall back to higher-D representation. Like a cascade classifier.

Or: different bottleneck dims for different tiles. Some tiles get 2D, others get 8D. Learned allocation.

**Verdict**: The critical experiment. This validates or invalidates the whole PQH thesis.

---

## Node 5: The Lookup-Only Model

**The Idea**: Eliminate all matrix multiplies. Pure routing + memory read.

**Practical Version**:
```python
class LookupOnlyFFN:
    codebook: [num_entries, d_model]  # The "knowledge"
    signatures: [num_entries, d_model]  # The "addresses"
    
    def forward(x):
        scores = x @ self.signatures.T
        idx = scores.argmax(dim=-1)  # Hard routing
        return self.codebook[idx]  # Pure lookup
```

That's... a 1-tile version. For multi-tile:
```python
def forward(x):
    idx = hierarchical_route(x)  # Returns tile index
    return self.tile_codebooks[idx][sub_route(x, idx)]
```

**Why This Might Work**:
- Ultimate sparsity: single memory read per input
- Training learns "what knowledge lives where"
- Inference is trivially parallelizable and cacheable

**Why This Might Fail**:
- Very limited expressiveness (can only output codebook entries)
- Soft blending probably necessary (then it's just attention over codebook)
- Discrete routing kills gradients

**Radical Extension**:
Hybrid: use lookup for "confident" routes, fall back to computation for "uncertain" routes. The model decides when to think vs. when to retrieve.

Or: hierarchical codebook. Top-level entries are themselves small codebooks. Recursive lookup.

**Verdict**: Philosophically pure but practically challenging. Research project, not near-term.

---

## Node 6: Soft Emergent Routing

**The Idea**: Instead of hard top-1 routing, do soft top-k with emergent (not learned) weights.

**Practical Version**:
```python
def soft_emergent_route(x, signatures, k=4, temperature=1.0):
    scores = x @ signatures.T
    topk_scores, topk_idx = scores.topk(k)
    weights = softmax(topk_scores / temperature)
    return topk_idx, weights

def forward(x):
    idx, weights = soft_emergent_route(x, self.signatures)
    outputs = [self.tiles[i](x) for i in idx]
    return (outputs * weights.unsqueeze(-1)).sum(dim=0)
```

**Why This Might Work**:
- Smooths the routing, might help gradient flow
- Still zero routing parameters (weights from signatures)
- Interpolates between tiles for ambiguous inputs

**Why This Might Fail**:
- More compute (k tiles instead of 1)
- Defeats the sparsity purpose
- Might converge to always using all tiles equally

**Radical Extension**:
Adaptive k. Easy inputs get k=1. Hard inputs get k=4. Temperature annealing during training: start soft, end hard.

Or: hierarchical soft routing. Soft at cluster level, hard at tile level. Best of both.

**Verdict**: Low risk, incremental improvement. Good for stabilizing training.

---

## Node 7: The Resonance Architecture

**The Idea**: Tiles communicate. The output of one tile influences routing to others.

**Practical Version**:
```python
def forward(x, n_iterations=3):
    h = x
    for _ in range(n_iterations):
        idx = route(h)
        h = x + self.tiles[idx](h)  # Residual
    return h
```

The routing is recalculated each iteration based on the updated hidden state.

**Why This Might Work**:
- Multi-step refinement (like diffusion models)
- Routing can "change its mind" as understanding deepens
- Allows complex computations via iteration

**Why This Might Fail**:
- Much more compute (n_iterations × single-pass)
- Might not converge (oscillation)
- Harder to train

**Radical Extension**:
Let iterations continue until routing stabilizes (fixed point). This is like a Hopfield network. The "answer" is the attractor state.

**Verdict**: Interesting for research. Connects to dynamical systems, attractor networks.

---

## Emerging Themes

Looking across all nodes, some patterns appear:

### Theme A: Levels of Routing
Multiple nodes involve routing at different granularities. There's something here about hierarchical content-addressing. The right architecture might have routing at every scale.

### Theme B: Compression as Constraint
The most radical ideas involve aggressive compression (2D bottleneck, ternary everything). Constraint can force efficiency, but too much kills expressiveness.

### Theme C: Soft vs Hard
The tension between differentiable (soft) and efficient (hard) appears everywhere. The answer is probably staged: soft for training, hard for inference.

### Theme D: Lookup vs Compute
The philosophical question: is intelligence computation or retrieval? The answer might be: both, with routing deciding which.

---

## Priority Stack

If I had to pick what to build:

1. **Node 4: Compression Funnel** - The critical experiment. If this works, everything else follows. If it fails, we learn why.

2. **Node 6: Soft Emergent Routing** - Low-risk improvement. Could help HybridKAN's routing entropy issue.

3. **Node 3: Ternary Splines** - The "maximum compression" experiment. Test how little precision we really need.

4. **Node 2: Fractal Routing** - The elegant architecture. Build after Node 4 validates the approach.

5. **Node 5: Lookup Only** - The moonshot. Pursue when other pieces are proven.

---

## What Wants to Emerge

The architecture that keeps appearing in my mind:

```
Input
  ↓
[Hierarchical Emergent Routing] ← from TriX (proven)
  ↓
[Tile with Compression Funnel] ← Node 4 (to prove)
  ↓
[Ternary Spline Lookup] ← Node 3 (for efficiency)
  ↓
[Expand + Residual]
  ↓
Output
```

This combines:
- Best routing (HierarchicalTriX, all tiles active)
- Aggressive compression (to make splines feasible)
- Maximal efficiency (ternary spline coefficients)

Let's call it: **SparseLookupFFN**

Next file: put on the engineering monocle and see if this actually makes sense.
