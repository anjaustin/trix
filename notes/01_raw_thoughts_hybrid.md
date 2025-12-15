# Raw Thoughts: Hybrid Possibilities

*Stream of consciousness. No filter. Let it flow.*

---

## Starting Point

We have two things that work equally well but differently:
- HierarchicalTriX: better routing, ternary weights, all tiles active
- HybridKAN: spline nonlinearity, bottleneck, slightly worse routing

The question isn't "which is better" - they tied. The question is: **what are they each doing right that the other isn't capturing?**

---

## What is routing really doing?

Routing is... addressing. Content-based addressing. The input says "I'm like this" and the system says "then you go here."

But wait. The *signature* is derived from *weights*. So the weights are encoding two things simultaneously:
1. What inputs they want (signature)
2. What to do with those inputs (computation)

That's elegant but also... constraining? The signature and the function are coupled. What if they shouldn't be?

What if we decouple them?

---

## Decoupling Signature from Computation

Idea: separate "what I want" from "what I do"

```
Tile {
  signature: what inputs route here (address)
  compute: what happens to those inputs (function)
}
```

Right now in TriX: signature = f(compute_weights)
What if: signature = learned_separately?

But then we lose the "zero routing parameters" property...

Unless... the signature comes from something ELSE in the tile. Like the spline coefficients. Or the bottleneck projection.

---

## The Spline as Address

Splines have this structure: grid cells with local coefficients.

Each grid cell is like... a mini-tile within the tile?

What if the spline's *structure* (which cells are "hot", which coefficients are large) defines the signature?

The spline would then be:
1. The computation (what it always was)
2. The address (emergent from its structure)

This feels circular but maybe productively circular?

---

## What is a spline really?

A spline is a piecewise linear function. It says: "in this region, do this linear thing."

That's... routing at a different level. The spline is routing within the tile.

So we have:
- Level 1: Route to cluster (coarse, based on signatures)
- Level 2: Route to tile (medium, based on signatures)  
- Level 3: Route to spline cell (fine, based on input values)

Three levels of routing. Each one is "content addressing" at a different granularity.

---

## The Fractal Idea

What if the whole thing is self-similar?

- A tile is a small model that routes to spline cells
- A cluster is a small model that routes to tiles
- The full model is a small model that routes to clusters

Each level has:
- Addresses (signatures / grid indices)
- Functions (weights / coefficients)
- Routing (similarity / binning)

The only difference is scale and what the "address" is derived from.

---

## Back to the Hybrid Question

Option 2 was: "Emergent Routing + Spline Tiles"

But maybe that's too simple. What if:

**Option 2b: Emergent Routing at EVERY level**

- Cluster level: signature from cluster aggregate
- Tile level: signature from tile weights (current)
- Spline level: signature from... spline coefficients?

How would spline-level emergent routing work?

Current spline: `idx = floor(x / cell_size)` - deterministic binning
Emergent spline: `idx = argmax(x @ cell_signatures)` - content-based

The cell signatures would be derived from the cell coefficients. Wild.

---

## The Ternary Spline

What if spline coefficients were ternary?

Traditional spline: `y = base + slope * x` where base, slope are floats
Ternary spline: `y = base + slope * x` where base, slope are in {-1, 0, +1}?

That's... very constrained. But maybe constraint is good?

With ternary coefficients:
- base: offset direction (-1, 0, +1)
- slope: rate of change direction (-1, 0, +1)

You'd need a scale factor. So: `y = scale * (base + slope * x)`

This would be maximally compressible. 2 bits for base, 2 bits for slope, plus one float scale per... cell? tile? layer?

---

## What about the input?

We've been thinking about compressing the weights/coefficients. What about the inputs?

HybridKAN does: `h = bottleneck(x)` before the spline. That's compressing the input.

What if we quantize the input too?

`h = bottleneck(x)` → `h_q = quantize(h)` → `spline(h_q)`

If h_q is 8-bit, the spline becomes a lookup table. 256 entries.
If h_q is 4-bit, it's 16 entries per dimension.

With a 128-dim bottleneck at 4-bit... that's still huge.

But what if h is only 2 dimensions? Then 4-bit gives us 16x16 = 256 entries total. That's the Spline2D case.

---

## The Compression Funnel

```
x (512-dim float)
  ↓ bottleneck
h (2-dim float)  
  ↓ quantize
h_q (2-dim, 16 levels each)
  ↓ spline lookup
y_q (from 256-entry table)
  ↓ expand
y (512-dim float)
```

This is aggressive. 512 dims → 2 dims → 256 discrete states.

The question is: does routing constrain x enough that this works?

If the tile only sees inputs that are "similar" (because routing), then maybe the within-tile variance is low. And 2 dimensions can capture it.

---

## The Information Bottleneck View

There's a theory: the information bottleneck. Compress the input to remove noise, keep signal.

Routing is a form of compression - it's selecting which "type" of input this is.

Then the bottleneck compresses further - what are the relevant dimensions for THIS type?

Then quantization compresses even more - what are the relevant levels?

Each stage removes information. The question is whether it removes noise or signal.

---

## What if routing IS the computation?

Radical thought: what if we don't need weights at all?

Current: `y = W @ x` (weights do computation)
Alternative: `y = codebook[route(x)]` (routing does everything)

The "computation" is just: figure out which entry you are, return that entry.

This is... vector quantization? VQ-VAE style?

The codebook would be the "knowledge." Routing would be the "retrieval."

"Don't learn what you can read" taken to the extreme: don't COMPUTE what you can LOOKUP.

---

## The Lookup-Only Model

Imagine a model where every layer is:
1. Route to an entry
2. Return that entry (or blend nearby entries)

No matrix multiplies. Just addressing and reading.

Training would learn:
1. How to route (the signatures)
2. What's at each address (the codebook entries)

Inference would be:
1. Compute route (dot products with signatures)
2. Read from memory

This is... associative memory. Content-addressable memory. What neural networks were before we got obsessed with backprop and gradients.

---

## Connecting Back to Splines

A spline IS a lookup with blending. You look up the cell, then blend based on position.

So "spline computation" is already "lookup + interpolation."

The question is: what's the right lookup granularity?

- Too fine (many cells): basically a dense matrix
- Too coarse (few cells): loses precision
- Just right: ???

Maybe the right granularity depends on the routing? If routing is coarse, cells can be fine. If routing is fine, cells can be coarse.

---

## The Adaptive Resolution Idea

What if spline resolution adapted to routing confidence?

High routing confidence → this tile really "owns" this input → coarse spline OK
Low routing confidence → input is ambiguous → need fine spline

This would be dynamic computation allocation. Certain inputs get more compute.

Like... attention? Attention allocates compute dynamically too.

---

## Attention as Routing

Wait. Attention is soft routing over positions. Each query routes to keys.

TriX routing is hard routing over tiles. Each input routes to one tile.

What if we made TriX routing soft? Route to top-k tiles, blend outputs?

That's... mixture of experts. MoE.

But MoE typically has a learned router. TriX's insight is emergent routing.

**Soft emergent routing**: route to top-k tiles based on signature similarity, blend by similarity score.

This might help the routing entropy issue in HybridKAN.

---

## I'm going in circles

Let me list the concrete ideas that have emerged:

1. Decouple signature from computation (separate "address" from "function")
2. Fractal routing (emergent routing at cluster, tile, AND spline cell level)
3. Ternary spline coefficients (maximally compressed)
4. Input quantization after bottleneck (enables true lookup)
5. Lookup-only model (no matmuls, just routing + reading)
6. Adaptive spline resolution (based on routing confidence)
7. Soft emergent routing (top-k blending)

Some of these are incremental. Some are radical. Let me sit with them.

---

## The one that keeps pulling me

The lookup-only model. "Don't compute what you can lookup."

It's philosophically aligned with TriX. It's the logical extreme.

But it feels impractical. How do you train it? Gradients through discrete routing?

Unless... you train with soft routing and deploy with hard routing. Soft-to-hard annealing.

Or... you use something like VQ-VAE's straight-through estimator. Discrete forward, continuous backward.

---

## Ending this stream

I need to let these ideas breathe. The next file will be about exploring the nodes of opportunity - which of these threads has legs?

Key tensions to resolve:
- Compression vs. expressiveness
- Discrete vs. differentiable  
- Simplicity vs. capability
- Elegance vs. practicality

The answer is probably not one extreme or the other. It's finding the right trade-off point.

But first, let's see what wants to emerge.
