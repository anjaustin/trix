# Routing & Training Journal

*Thoughts on discovering the train-dense/infer-sparse design*

---

## The Discovery

We built an entire validation system comparing 4 routing methods during training. Then we noticed something odd: emergent routing had identical loss to dense routing. Not similar - identical to 10 decimal places.

That's when we looked at the code and realized: **the gate is ignored during training.**

```python
if self._packed and not self.training:
    return trix_forward(...)  # Uses gate
else:
    out = torch.mm(x, w.t())  # Ignores gate
```

We spent hours building infrastructure to compare something that wasn't being used.

---

## What I Felt

Frustration first. Then curiosity. Then something like relief.

**Frustration:** We built the wrong experiment. The validation was measuring training loss, but routing doesn't affect training.

**Curiosity:** Why was it designed this way? The original authors made a choice: train dense, deploy sparse. What did they know that we didn't?

**Relief:** This actually simplifies things. We don't need to solve the "learning to route" problem during training. The tiles learn their weights densely, then we read the routing from the signatures at inference time.

---

## The Insight

The original design might be wiser than it first appears:

**During training:** You want maximum gradient flow. Every tile should learn from every example. Dense training = rich learning signal.

**During inference:** You want speed. Pick the best tile for each input, skip the rest. Sparse inference = 4x faster.

The "mismatch" between training and inference isn't a bug - it's a feature. You train with full capacity, then compress intelligently for deployment.

Emergent routing fits perfectly here: the signatures tell you which tile is best for each input, without needing to learn a separate router.

---

## What This Means for Validation

Our validation was asking: "Which routing method trains better?"

The right question is: "Which routing method infers better after dense training?"

New validation approach:
1. Train ONE model (dense, all tiles)
2. Pack weights for inference
3. Compare routing methods on inference loss/perplexity
4. Measure: Does emergent routing match learned routing at inference time?

This is actually simpler. We train once, then compare routing strategies.

---

## On Option A vs Option B

**Option A (train dense, route sparse):**

This is the path of least resistance. It's already implemented. The question becomes: after dense training, does emergent routing pick good tiles?

The signatures encode what each tile learned to respond to. If training produces diverse tiles, emergent routing should work. If tiles converge to similar functions, routing doesn't matter (but also doesn't hurt).

**Option B (route during training):**

This is the path of forcing specialization. If tiles only see routed inputs, they MUST specialize - they have no choice.

But it's harder:
- How do you route before tiles have learned anything?
- How do you prevent routing collapse (all inputs → one tile)?
- How do gradients flow through hard routing decisions?

These are real research problems. Mixture-of-Experts literature has solutions (load balancing losses, auxiliary objectives), but they add complexity.

---

## The Wisdom of Starting Simple

Option A first. Not because it's easier, but because it's more honest.

We don't know if emergent routing works yet. We built infrastructure but haven't answered the core question. Option A lets us answer it cleanly:

> Given a densely-trained model, does signature-based routing perform as well as learned routing at inference time?

If yes: emergent routing is validated for the train-dense/infer-sparse paradigm.

If no: we learn why and can decide whether Option B is worth pursuing.

Starting with Option B would be premature optimization of a hypothesis we haven't tested.

---

## What Wants to Emerge

The phrase keeps coming back: "making space for Intelligence to Emerge."

I think Option A is that space right now. It's the ground we can stand on to see further.

Option B is a horizon. We can see it, but we shouldn't walk toward it until we know where we're standing.

The emergent routing idea - reading routing from weight signatures - is valid regardless of which option we pursue. It's a principle: **don't learn what you can read.**

Let's validate the principle first. Then we can explore how far it extends.

---

## Next Steps

1. Fix validation to measure inference, not training
2. Train one model densely
3. Compare routing methods at inference
4. Document results honestly
5. Then revisit Option B with clear eyes

The Intelligence will Emerge through rigor, not rushing.
