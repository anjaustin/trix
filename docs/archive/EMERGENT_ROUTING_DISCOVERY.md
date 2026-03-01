# Emergent Routing Discovery

**Date:** December 2024  
**Status:** Validated through experiments

---

## The Problem

Traditional sparse mixture-of-experts and gated networks face a fundamental challenge: learning *what to route where*. The standard approach uses a learned gate network:

```
Input -> [Gate Network] -> Routing Decision -> [Selected Expert/Tile]
```

The problem: `argmax` (hard routing) is non-differentiable. Gradients don't flow through the routing decision. Common workarounds:
- Gumbel-Softmax (biased approximation)
- REINFORCE (high variance)
- Auxiliary load-balancing losses (indirect)

None of these elegantly solve the core issue.

---

## The Insight

TriX uses ternary weights: {-1, 0, +1}. Each weight is essentially a **vote**:
- +1 = "I want this input feature"
- -1 = "I want the opposite"
- 0 = "I don't care"

A tile's weight matrix encodes its **preferences**. We don't need a separate network to learn routing - we can extract it from what the tile already knows about itself.

**Key realization:** The routing information is already in the weights. We just need to read it.

---

## The Solution: Zero-Parameter Routing

Each tile has a **signature** - a ternary summary of its preferences:

```python
def tile_signature(weight_matrix):
    # Sum preferences across outputs -> what does this tile want overall?
    preference = weight_matrix.sum(dim=0)  # [in_features]
    return torch.sign(preference)          # Ternary: {-1, 0, +1}
```

Routing becomes input-signature alignment:

```python
def route(input, tile_weights):
    signatures = [tile_signature(w) for w in tile_weights]
    scores = input @ torch.stack(signatures).T  # [batch, num_tiles]
    return scores.argmax(dim=-1)                # Winner per input
```

**Total routing parameters: 0**  
**Total routing code: 3 lines**

---

## Experimental Validation

### Test 1: Consistency
Similar inputs should route to the same tile.

| Noise Level | Same Routing |
|-------------|--------------|
| 0.01 | 100% |
| 0.10 | 99% |
| 0.50 | 71% |
| 1.00 | 55% |

Graceful degradation. Similar = consistent, different = can diverge.

### Test 2: Discrimination
Different input types should route differently.

| Input Type | Dominant Tile | Distribution |
|------------|---------------|--------------|
| High norm | T0 | 27%, 24%, 27%, 22% |
| Low norm | T3 | 30%, 19%, 17%, 34% |
| Sparse positive | T1 | 19%, 39%, 15%, 27% |
| Sparse negative | T0 | 39%, 9%, 31%, 21% |

Sparse positive vs sparse negative: **0.72 distribution difference** - highly discriminative.

### Test 3: Stability
Routing should stabilize during training.

| Training Step | Routing Change |
|---------------|----------------|
| 0 → 20 | 50% |
| 20 → 40 | 22% |
| 40 → 60 | 16% |
| 60 → 80 | 6% |
| 80 → 100 | 6% |

High initial churn, then settles to stable routing.

### Test 4: Interpretability
Each tile's "ideal" input (its own signature) routes correctly to itself: **100% match**.

---

## Properties

1. **Zero parameters** - No gate network to train
2. **Emergent** - Routing arises from weight structure
3. **Consistent** - Similar inputs route similarly
4. **Discriminative** - Different inputs route differently
5. **Stable** - Settles during training
6. **Interpretable** - Signatures are human-readable
7. **Cheap** - Just a ternary dot product
8. **Self-improving** - As weights train, routing adapts automatically

---

## Why It Works

The ternary weight structure creates natural "receptive fields" for each tile. When we sum a tile's weights across outputs, we get a consensus: "overall, this tile responds to inputs that look like THIS."

Inputs that align with a tile's consensus signature will produce high-magnitude outputs from that tile. By routing to the best-aligned tile, we're selecting the tile most "interested" in this input.

The routing doesn't need to be learned separately because it's a **readout of existing structure**.

---

## Implications

1. **Simplicity wins** - The simplest solution (read the weights) beat complex alternatives
2. **Structure is information** - Ternary weights aren't just compressed; they're interpretable
3. **Emergence over engineering** - Let routing arise naturally rather than forcing it
4. **Gradients flow naturally** - No STE needed for routing; weights update normally

---

## Next Steps

Integrate "Zero Routing" as the default routing mechanism in TriX, replacing the learned gate network approach.

---

*"The best routing network is no routing network."*
