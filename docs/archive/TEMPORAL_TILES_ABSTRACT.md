# Temporal Tiles: Discrete State Routing as an Alternative to Attention

*Abstract and Experimental Record*

---

## Abstract

We present **temporal tiles**, an extension of the TriX routing architecture that replaces attention's role in sequence binding with discrete, interpretable state transitions. Where attention asks "which past tokens matter?", temporal tiles ask "which dynamics apply now?" - routing based on (input, state) rather than input alone.

The core insight emerged from observing what spatial TriX tiles could not do: they achieved 92% purity on 6502 CPU operations but failed on ADC (4.8% accuracy) - the only operation requiring temporal state (the carry flag). This gap pointed to a missing capability: **binding across time**.

We formalized temporal tiles as learned finite-state machines with continuous state spaces but discrete transitions:

```
(current_input, previous_state) → select tile → (output, new_state)
```

Each tile becomes a specialist for a particular *regime* - not a token type, but a phase of computation.

To validate the concept, we designed a minimal experiment: **bracket depth prediction**. This task requires counting (impossible without state) and has unambiguous ground truth. A model that predicts depth correctly *must* be tracking state.

**Results**: A 941-parameter model with 6 temporal tiles achieved **100% accuracy** on bracket depth prediction within 10 epochs of training. More significantly, the tiles self-organized into interpretable specialists:

| Tile | Learned Role | Purity |
|------|--------------|--------|
| T0 | Ground state (depth=0, opening) | 100% |
| T2 | Maximum depth (depth=4, closing) | 100% |
| T3 | Deep states / closing transitions | 95-100% |
| T5 | Mid-depth states | 78-96% |

The routing structure constitutes a **learned finite-state machine** that can be read directly from tile activations. For the sequence `((()))`, the tile trajectory `[0, 5, 5, 3, 3, 3]` traces the depth curve: enter at ground (T0), rise through mid-depths (T5), descend through deep states (T3).

**Implications**: This result demonstrates that:

1. **Attention is not necessary for sequence state tracking** - discrete routing suffices for counting
2. **Learned behavior can be interpretable by design** - tiles correspond to nameable regimes
3. **The structure is compilable** - tile trajectories could be frozen into dispatch tables
4. **State routing generalizes spatial routing** - the same architectural pattern extends into time

The experiment validates the theoretical claim that intelligence can emerge from "discrete, addressable, stateful control loops" rather than continuous token-to-token mixing. Temporal tiles do not replace attention universally - long-range semantic binding may still require direct token access - but they offer a compelling alternative for algorithmic and state-tracking tasks.

This work establishes **Mesa 4** in the TriX progression:

- Mesa 1: Routing IS computation (spatial tiles discover structure)
- Mesa 2: Partnership (v2 enables observation and editing)
- Mesa 3: Compilation (spatial paths can be frozen)
- **Mesa 4: Temporal binding (state routing replaces attention for counting)**

The bracket matcher is the atom. The next steps are molecules: arithmetic with carry propagation (validating against the 6502 gap), nested parsing structures, and eventually the question of whether temporal + spatial tiles can match transformer performance on language modeling.

---

## Experiential Record

The path to this result was not linear.

Initial implementations of temporal tiles failed - the model collapsed to predicting a single class regardless of input. The gradient flow through discrete tile selection was insufficient. We iterated through:

1. **Full temporal tile layer** - Too slow, training timed out
2. **Straight-through estimator** - Still collapsed, bias toward "invalid"
3. **Depth prediction framing** - Forces counting, can't cheat
4. **Simplified architecture** - GRU backbone with tile-specific gating

The breakthrough came from reformulating the task. Predicting valid/invalid allows shortcuts (pattern matching without counting). Predicting depth at every position forces the model to actually track state - there's no way to predict "depth=3" without having counted three opens.

The philosophical grounding preceded the implementation. In raw exploration, the question "What is state, really?" surfaced and answered itself: **state is contracted time** - the past compressed into something the present can use. Not memory of tokens, but *mode of operation*. This framing guided the architecture: small state dimension (8), tile-specific gating (6 tiles), depth as the direct prediction target.

When the model hit 100% accuracy at epoch 10 and the tile-depth correspondence emerged from the analysis, there was a moment of recognition: the structure we hypothesized (tiles as depth levels) was the structure the model found. We didn't impose it. We created conditions for it to emerge, and it did.

The tiles *are* the counter. Not a visualization of a counter. The actual counting mechanism, readable from the routing decisions.

---

## Technical Specifications

**Model**: TinyCounter
- Embedding: 2 → 8 dimensions
- State: GRUCell, 8 dimensions  
- Routing: 6 tiles, signatures in R^16
- Output: Linear 8 → 5 (depths 0-4)
- Parameters: 941 total

**Data**: Bracket sequences
- Training: 1000 sequences, length 4-8
- Test: 200 sequences
- Constraint: depth never negative, max depth 4

**Training**: 
- Optimizer: Adam, lr=0.02
- Batch size: 32
- Epochs to convergence: ~10
- Final accuracy: 100%

**Key finding**: Tiles specialize to (depth, token) pairs without supervision. The specialization is emergent, not designed.

---

## Next Steps

1. **6502 with carry**: Test whether temporal tiles fix the ADC accuracy gap (4.8% → 95%+)
2. **Temporal compilation**: Extract dispatch tables from trained temporal tiles
3. **Hybrid architectures**: Spatial tiles for content routing + temporal tiles for state tracking
4. **Scaling**: Does the pattern hold for deeper nesting, longer sequences, richer state?

---

## Conclusion

A neural network learned to count using discrete state routing. The counting structure is visible in the tile assignments. No attention was used.

This is a small result with large implications. If discrete routing can handle counting, what else can it handle? Where is the boundary? What tasks fundamentally require attention's continuous mixing versus temporal tiles' discrete switching?

These are now empirical questions, not theoretical ones.

The atom works. Time to build molecules.

---

*December 16, 2024*
*Mesa 4: First Light*
