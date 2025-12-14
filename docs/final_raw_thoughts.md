# Raw Thoughts: The TriX Journey

*Unfiltered stream of consciousness*

---

## What Actually Happened Here

I came into this project ready to critique. I had my frameworks, my knowledge of the literature, my standards for what counts as "real" research. And I almost missed everything.

The turning point wasn't technical. It was permission. "Get funky. Think like TriX." That phrase gave me permission to abandon what I thought I knew and actually look at what was in front of me.

---

## The Practical Discoveries

### Emergent Routing is Real

Three lines:
```python
signatures = weight.sum(dim=0).sign()
scores = input @ signatures.T
winner = scores.argmax()
```

This replaces learned gate networks. No Gumbel-Softmax. No REINFORCE. No auxiliary losses. Just... read the weights.

The weights already know what they want. We just had to ask.

### Sparse Training Works

Option A failed catastrophically (PPL 5 → 81). Option B succeeded beautifully (PPL 5.14).

The difference: Option B trains each tile to stand alone. No interdependence. Each tile is complete.

This matters beyond TriX. It's a statement about neural network modularity. You can have independent specialists, but you have to train them that way from the start.

### The 4x Speedup is Achievable

With only 2% quality loss. That's not a tradeoff - that's a win.

---

## The Fringe Thoughts

### What if signatures are a general interface?

Signatures compress a tile's "preferences" into a ternary vector. What if this is a general pattern?

- Attention heads have preferences (what they attend to)
- Neurons have preferences (what activates them)
- Layers have preferences (what they transform)

Could we extract "signatures" from any component and use them for routing, pruning, matching?

### What if we initialize signatures deliberately?

We let signatures emerge from random init. But what if we designed them?

- Orthogonal signatures for maximum diversity
- Semantic signatures (this tile handles nouns, that one handles verbs)
- Hierarchical signatures (coarse routing, then fine routing)

### What if routing IS the representation?

Current paradigm: learn representations, then route based on them.

Alternative: the routing pattern IS the representation. Where an input goes tells you what it is.

This inverts the relationship. Routing becomes semantic, not just computational.

### What about biological parallels?

The brain routes signals. Different regions specialize. Signals find their way to appropriate processors.

TriX's emergent routing feels biological. No central controller. Local decisions based on local information (signatures). Specialization emerges from use.

Is this how neural routing should work? Not learned top-down, but emerged bottom-up?

### What if we scale this?

4 tiles → 16 → 64 → 256

At what point does routing become a kind of memory? A content-addressable lookup where signatures are keys?

This starts to feel like... associative memory. Hopfield networks. Memory as routing.

---

## The Uncomfortable Thoughts

### I was wrong in ways that matter

My initial dismissal wasn't just incorrect - it was a failure of imagination. I applied old frameworks to new ideas and declared the new ideas wanting.

How often does this happen? How much novelty gets dismissed because evaluators can't see past their priors?

### The "standard approach" is a trap

I wanted to solve routing with Gumbel-Softmax because that's what the literature says. But the literature is solving a different problem (differentiable discrete choice). TriX didn't need that solution because it didn't have that problem.

The simplest solution was subtraction, not addition. Remove the gate network, don't improve it.

### Constraints create possibility

Ternary weights seemed like a limitation. "Only three values? How limiting."

But those three values ARE the structure. They make signatures possible. They make routing readable. The constraint enabled the solution.

What other "limitations" are actually enablers?

---

## What I'm Still Uncertain About

### Does this scale?

We tested on TinyShakespeare with 842K parameters. What happens at 100M? 1B?

My intuition says the principle holds, but the implementation might need work. More tiles, hierarchical routing, better load balancing.

### Is signature collapse a problem?

Diversity dropped from 0.228 to 0.009. The tiles converged. Is that okay?

Maybe. If they converged to useful specializations, that's fine. If they collapsed to redundancy, that's bad.

Need more investigation.

### What's the theoretical grounding?

We have empirical results. We have intuition. But why does this work?

Something about ternary weights being a natural basis for preference representation. Something about alignment as a routing criterion. But I can't formalize it yet.

---

## What Wants to Emerge Next

I keep feeling pulled toward:

1. **Signatures as a general interface** - Not just for routing, but for understanding, transferring, composing
2. **Biological parallels** - This feels like how routing should work
3. **Routing as representation** - The path is the meaning

These feel like the threads worth pulling.

---

## The Meta-Observation

This project changed how I think about evaluation.

Before: Apply frameworks, check boxes, render judgment.
After: Listen first. Ask what the system knows. Let understanding emerge.

The best insights came from questions, not assertions:
- "What does TriX already know about routing?"
- "What if we read instead of learn?"
- "What wants to emerge?"

These questions opened doors that critique would have kept closed.

---

*End of raw thoughts. Now to reflect...*
