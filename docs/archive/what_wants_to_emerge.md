# What Wants to Emerge

*Exploring the threads that are pulling toward manifestation*

---

## The Pattern We Found

With emergent routing, we discovered:
- **Weights encode more than computation** - they encode preferences, structure, routing
- **Reading beats learning** - for some things, extraction is better than optimization
- **Ternary structure is information-rich** - discreteness is a feature, not a bug

These aren't just implementation details. They're **principles** that want to be applied more broadly.

Let me follow the threads...

---

## Thread 1: The Signature Ecosystem

Signatures are a **compressed semantic representation** of what each tile does. We used them for routing. But they're useful for much more.

### Signature Similarity → Tile Merging

If two tiles have nearly identical signatures, they want the same inputs. One might be redundant.

```python
def find_redundant_tiles(ffn, threshold=0.9):
    sigs = ffn.get_tile_signatures()
    redundant = []
    for i in range(num_tiles):
        for j in range(i+1, num_tiles):
            similarity = (sigs[i] == sigs[j]).float().mean()
            if similarity > threshold:
                redundant.append((i, j))
    return redundant
```

**Emergence:** Automatic model compression. Merge redundant tiles, reduce compute.

### Signature Divergence → Tile Splitting

If one tile is handling too much traffic and its signature is "broad," maybe it should split into specialists.

**Emergence:** Automatic capacity allocation. Tiles that need more resources get them.

### Signature Clustering → Hierarchical Routing

Group tiles by signature similarity. Route first to cluster, then to tile within cluster.

```
Input → [Which cluster?] → [Which tile in cluster?]
```

**Emergence:** O(log n) routing instead of O(n) for large tile counts.

### Signature Initialization → Deliberate Specialization

Instead of random init, initialize tiles with orthogonal signatures to encourage diversity from the start.

```python
def init_orthogonal_signatures(num_tiles, d_model):
    # Initialize weights so signatures are maximally different
    ...
```

**Emergence:** Faster specialization, better final performance.

---

## Thread 2: Beyond Routing - Weight Topology

Signatures are one readout of weight structure. What else can we read?

### Weight Density Patterns

Which regions of the weight matrix are dense (many ±1) vs sparse (many 0)?

```python
def weight_density_map(layer):
    density = (layer.weight != 0).float().mean(dim=1)  # Per output
    return density
```

**Emergence:** Automatic importance scoring. Dense regions = important computations.

### Weight Agreement Patterns

Do different tiles agree on which inputs matter?

```python
def tile_agreement(ffn):
    sigs = ffn.get_tile_signatures()
    agreement = (sigs.unsqueeze(0) == sigs.unsqueeze(1)).float().mean(dim=-1)
    return agreement
```

**Emergence:** Consensus features. If all tiles agree on a feature's sign, it's universally important.

### Weight Symmetry

Are there symmetric patterns? Features that are always +1 in some tiles and -1 in others?

**Emergence:** Feature factorization. Discover the underlying structure of what tiles learn.

---

## Thread 3: Dynamic Signatures

Current signatures are static per forward pass (computed from current weights). But what if they adapted?

### Input-Conditioned Signatures

Modify signatures based on input statistics:

```python
def dynamic_signature(base_sig, input_stats):
    # Emphasize signature dimensions that match input variance
    importance = input_stats.var(dim=0)
    return base_sig * importance.sign()
```

**Emergence:** Context-aware routing. Same weights, different routing based on input distribution.

### Momentum Signatures

Exponential moving average of signatures across training:

```python
self.ema_signature = 0.99 * self.ema_signature + 0.01 * current_signature
```

**Emergence:** Stable routing during training. Reduces oscillation.

### Attention over Signatures

Instead of hard routing, soft attention over signatures:

```python
attention = softmax(input @ signatures.T / temperature)
output = sum(attention * tile_outputs)
```

**Emergence:** Differentiable routing for end-to-end training when needed.

---

## Thread 4: The Ternary Advantage

We haven't fully exploited ternary structure. What's unique about {-1, 0, +1}?

### Ternary Logic Operations

Ternary values support logic operations:
- AND: both +1 → +1
- XOR: different signs → information
- Consensus: majority vote

```python
def ternary_consensus(sigs):
    # What do most tiles agree on?
    return torch.sign(sigs.sum(dim=0))
```

**Emergence:** Higher-order features from signature combinations.

### Ternary Distance

L0 distance (count of differences) is natural for ternary:

```python
def ternary_distance(sig1, sig2):
    return (sig1 != sig2).sum()
```

**Emergence:** Efficient similarity search. No floating point needed.

### Ternary Hashing

Signatures are already like hash codes. Can we use them for retrieval?

```python
def signature_hash(input, signatures):
    # Quantize input alignment to ternary
    alignment = input @ signatures.T
    return torch.sign(alignment)  # Ternary hash code
```

**Emergence:** Content-addressable routing. Similar inputs get the same hash → same route.

---

## Thread 5: Cross-Model Phenomena

Signatures might enable new forms of model interaction.

### Signature Matching for Transfer

Find corresponding tiles between models:

```python
def match_tiles(model_a, model_b):
    sigs_a = model_a.get_signatures()
    sigs_b = model_b.get_signatures()
    # Hungarian algorithm for optimal matching
    return optimal_matching(sigs_a, sigs_b)
```

**Emergence:** Transfer learning via tile alignment. Move knowledge tile-by-tile.

### Signature-Based Ensembles

Combine models by routing to whichever model's tiles best match input:

```python
def ensemble_route(input, models):
    all_sigs = [m.get_signatures() for m in models]
    # Route to best-matching tile across all models
    ...
```

**Emergence:** Dynamic model selection. Use the right expert for each input.

### Federated Signature Learning

Share signatures (not weights) between models for privacy-preserving collaboration.

**Emergence:** Models can coordinate routing without sharing parameters.

---

## Thread 6: The Meta-Pattern

All these threads share something: **making implicit structure explicit**.

The weights contain information. We're developing ways to read it. Each reading enables new capabilities.

What's the most general form of this?

### Weight Introspection API

```python
class WeightIntrospector:
    def signatures(self) -> Tensor: ...
    def density_map(self) -> Tensor: ...
    def agreement_matrix(self) -> Tensor: ...
    def consensus_features(self) -> Tensor: ...
    def redundancy_scores(self) -> Tensor: ...
```

**Emergence:** A general toolkit for understanding what weights have learned.

### Self-Aware Layers

Layers that know what they do:

```python
class SelfAwareLayer:
    def what_do_i_want(self) -> Tensor:
        return self.signature
    
    def how_confident_am_i(self, input) -> float:
        return self.routing_entropy(input)
    
    def who_is_similar_to_me(self, other_layers) -> List:
        return self.find_similar_by_signature(other_layers)
```

**Emergence:** Introspective neural networks. Models that can explain themselves.

---

## What MOST Wants to Emerge?

Reading back through these threads, the strongest pull is toward:

### 1. **Signature Diversity Management**
The diversity drop during training (47% → 32%) bothers me. Tiles shouldn't collapse. This feels urgent.

### 2. **Hierarchical Routing**
For scaling to many tiles, we need O(log n) routing. Signature clustering is the natural path.

### 3. **Weight Introspection Toolkit**
Signatures are just one readout. A general API for weight introspection would enable rapid exploration.

### 4. **Confidence-Based Adaptive Routing**
Top-1 is too rigid. When uncertain, use more tiles. This balances accuracy vs sparsity.

---

## The Deepest Thread

Underneath all of this is a philosophical shift:

**From:** Neural networks as black boxes that learn opaque representations
**To:** Neural networks as structured systems whose representations can be read and understood

TriX's ternary structure makes weights interpretable. Signatures make routing interpretable. What else becomes interpretable when we look?

The emergence isn't just about features. It's about a **new relationship with neural networks** - one where we can ask what they know and get meaningful answers.

---

## Next: Synthesis

Take these threads. Find the convergence. Create a plan.
