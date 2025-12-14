# Option B Plan: Train Sparse

*The path to true sparse neural networks with emergent routing*

---

## Why Option B?

Option A failed: Dense training + Sparse inference = PPL 81 (vs PPL 5 dense).

**Option B:** Train sparse from the start. Each tile learns to stand alone.

---

## Core Design

```python
for batch in data:
    # 1. Compute routing BEFORE forward
    gates = compute_emergent_routing(batch)
    
    # 2. Forward with gating (only routed tile computes)
    output = model(batch, gates, use_gate=True)
    
    # 3. Gradients only flow to active tiles
    loss.backward()
```

---

## Challenges & Solutions

### 1. Bootstrap Problem
Random signatures still partition input space. Emergent routing works from init.

### 2. Tile Collapse  
Add load balancing loss: `balance_loss = usage.std() / usage.mean()`

### 3. Data Efficiency
Each tile sees ~25% of data. Solution: longer training, larger batches.

### 4. Routing Stability
Use signature EMA and/or routing temperature annealing.

---

## Success Criteria

1. Sparse-infer loss within 20% of dense-infer
2. >3x inference speedup
3. Stable training without collapse
4. Meaningful tile specialization

---

## Timeline

~12 hours: Implement, train, analyze.

*Ready to build when you are.*
