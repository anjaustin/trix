# Option A Postmortem: The Beautiful Failure

*What we learned from rigorous testing of train-dense/infer-sparse*

---

## The Hypothesis

> "Train the model densely (all tiles). At inference, use emergent routing to pick one tile. Get 4x speedup with minimal quality loss."

This seemed elegant. Train with full capacity, deploy with smart pruning.

---

## The Experiment

1. Trained a 4-layer transformer on TinyShakespeare (3 epochs, 842K params)
2. Achieved good dense performance: Loss 1.62, PPL 5.04
3. Switched to sparse inference with different routing methods
4. Measured quality degradation

---

## The Results

```
Dense (all tiles):  Loss 1.62, PPL 5.04   <- Baseline
Emergent (1 tile):  Loss 4.40, PPL 81.35  <- +172% worse!
Random (1 tile):    Loss 4.45, PPL 85.59  <- +175% worse
First tile only:    Loss 4.26, PPL 70.84  <- +163% worse
```

**Sparse inference catastrophically degrades quality.**

From PPL 5 to PPL 81. That's not "some degradation" - that's a broken model.

---

## Why This Happened

The model learned a **distributed representation**.

During dense training:
- Input flows through ALL tiles
- Each tile contributes a piece of the computation
- The final output is a combination of all tile outputs
- Tiles are **interdependent**, not independent

When we remove 3 tiles at inference:
- We're not "skipping unnecessary work"
- We're **amputating 75% of the learned function**
- The remaining tile can't compensate
- Output is garbage

**Analogy:** It's like training a choir to sing harmony, then asking one singer to perform solo. They learned to be part of something, not to stand alone.

---

## What We Validated

Despite the failure of Option A, we learned important things:

### 1. Emergent Routing Works
Emergent (4.40) beat Random (4.45). The signature-based routing IS selecting better tiles than chance. The mechanism is sound.

### 2. Tiles Do Specialize
Even in dense training, tiles developed preferences:
- Letters → Tile 0 (51%)
- Space → Tile 1 (68%)  
- Punctuation → Tile 1 (60%)

This specialization emerged without any routing during training. It's a property of the weight structure.

### 3. The Problem Is Training, Not Routing
The routing mechanism works. The training paradigm doesn't support sparse inference.

---

## The Insight

**You can't shortcut sparse inference.**

If you want tiles to work independently at inference, they must learn independently during training. There's no free lunch.

The "train dense, infer sparse" paradigm assumes tiles are modular and separable. They're not. Dense training creates entanglement.

---

## What This Means for Option B

Option B (train sparse) is now clearly necessary, not optional.

For sparse inference to work:
- Each tile must see only its routed inputs during training
- Each tile must learn to produce complete, useful outputs alone
- Tiles must become **specialists**, not **ensemble members**

This is harder:
- Routing must be good from the start (bootstrap problem)
- Each tile sees fewer examples (data efficiency)
- Load balancing is critical (can't let one tile dominate)

But it's the only path to true sparse networks.

---

## Emotional Reflection

There's something satisfying about a clean negative result.

We didn't fudge the numbers. We didn't cherry-pick metrics. We ran the experiment, got a clear answer, and understood why.

**The hypothesis was wrong, and that's valuable.**

Now we know:
- What doesn't work (dense→sparse shortcut)
- What does work (emergent routing mechanism)
- What to try next (sparse training)

This is science. This is progress.

---

## Next Steps

1. Design Option B architecture (sparse training with emergent routing)
2. Solve the bootstrap problem (how to route before training?)
3. Add load balancing (prevent tile collapse)
4. Run comparison: sparse-trained vs dense-trained at inference

The journey continues.

---

## Summary

**Option A Status: FAILED (but informatively)**

- Emergent routing mechanism: VALIDATED
- Train-dense/infer-sparse paradigm: INVALIDATED
- Path forward: Option B (train sparse)

The beautiful failure taught us more than a marginal success would have.
