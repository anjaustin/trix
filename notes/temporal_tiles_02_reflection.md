# Temporal Tiles: Reflection
*Extracting the nodes. What actually emerged?*

---

## Node 1: State as Contracted Time

The raw insight: **State is the past compressed into something the present can use.**

Attention keeps all past tokens accessible and mixes softly. Temporal tiles would compress the past into a regime label: "I am in THIS mode now."

This is a fundamentally different information architecture:
- Attention: O(n) past tokens, soft weights
- Temporal tiles: O(1) state vector, hard regime

The question isn't "which past tokens matter?" but "which DYNAMICS apply now?"

**Implication**: State dimensionality should be small. Not a full representation of the past, but a MODE SELECTOR. 16-64 dims, not 512.

---

## Node 2: Soft States, Hard Transitions

The hybrid that keeps appearing: **continuous state space, discrete transitions**.

Classical FSMs: discrete states, discrete transitions → designed by hand, limited expressiveness

RNNs: continuous states, continuous transitions → learned, but uninterpretable soup

Temporal tiles: continuous states, discrete transitions → learned AND interpretable

The discrete transition (tile selection) creates BOUNDARIES. Observable moments. "The system switched modes HERE."

**Implication**: Interpretability comes from the hard edges, not the soft interiors. The tile selection IS the explanation.

---

## Node 3: The Carry Flag as Proof of Need

ADC at 4.8% while everything else hit 95%+. This wasn't a bug - it was a SIGNAL.

ADC is the only 6502 operation where previous state affects current output. The carry flag from operation N determines the result of operation N+1.

We patched it with input encoding (soroban). But the real fix is temporal state:
- "Carry is set" = a regime
- In this regime, ADC behaves differently
- The regime persists until cleared

The 6502 already tells us: **some computations require temporal binding**. We just weren't listening.

**Implication**: The 6502 is the perfect test bed. We already have the ground truth. We just need to add state.

---

## Node 4: Bracket Matching as Minimal Test

The simplest task that requires counting:

```
( ( ) ) → valid (depth 0→1→2→1→0)
( ) ) → invalid (depth goes negative)
```

Flat routing can't count. Attention can (expensively). Temporal tiles should be able to with ~4 tiles implementing a counter.

This is the ATOM of temporal capability:
- If 4 tiles can count parentheses, the thesis holds
- If they can't, we learn exactly what's missing

**Implication**: Don't start with language modeling. Start with bracket matching. Prove counting first.

---

## Node 5: Temporal Dispatch Tables

The compilation insight extends naturally:

Spatial: `class → tile`
Temporal: `(regime, class) → (tile, next_regime)`

That second form is literally a state transition table. An FSM definition. A PROGRAM.

```
DEPTH_0, ( → tile_inc, DEPTH_1
DEPTH_0, ) → tile_err, ERROR
DEPTH_1, ( → tile_inc, DEPTH_2
DEPTH_1, ) → tile_dec, DEPTH_0
```

This isn't interpretation of a neural network. This is EXTRACTION of a program. The learned behavior becomes readable, versionable, diffable.

**Implication**: If spatial compilation gave us "git for routing," temporal compilation gives us "git for control flow."

---

## Node 6: The Attention Replacement Thesis

Attention serves four roles:
1. Conditional compute (what runs)
2. Addressability (which specialist)
3. Sequence binding (connecting past to present)
4. Credit assignment (learning what mattered)

Spatial tiles handle 1 and 2. Temporal tiles handle 3. Compilation handles 4 (by making decisions explicit and editable).

If all four can be covered by discrete routing + state:
- Attention becomes optional
- It's one implementation, not the only one
- The discrete alternative is more interpretable

**Implication**: We're not optimizing transformers. We're proposing an alternative computational substrate.

---

## Node 7: The Long-Range Question

The honest uncertainty: **Can discrete state handle long-range dependencies?**

Attention binds token 1 to token 1000 directly. State must carry information through 999 steps. Can it survive?

Possible answers:
1. Yes, if state is structured right (counters, flags, registers)
2. No, some tasks fundamentally need direct binding
3. Hybrid: state for local/algorithmic, attention for long-range/semantic

This is the EMPIRICAL question. We don't know until we test.

**Implication**: Design experiments that probe range. Bracket matching (short). Arithmetic (medium). Language (long). See where it breaks.

---

## Node 8: Cognitive Glue

The Levin frame: cells coordinate through local rules and state, not central control. Patterns emerge.

Temporal tiles as cognitive glue:
- Each position inherits state from previous
- Each position routes based on (input, state)
- Global structure emerges from local rules

This is a different model of intelligence:
- Not "look at everything and decide" (attention)
- Not "flow information smoothly" (RNN)
- But "switch modes discretely, let patterns emerge"

**Implication**: The right metaphor isn't "memory" but "phase." The system is in phases. Transitions are discrete events. Behavior depends on phase.

---

## Node 9: Surgery Through Time

If spatial surgery lets you edit WHAT computes, temporal surgery lets you edit WHEN.

- Insert a new regime: "Handle backticks like quotes"
- Freeze a transition: "Always go from DEPTH_2 to ERROR on overflow"
- Modify a tile's state update: "This tile should also set the carry flag"

The system becomes programmable in time, not just space.

**Implication**: This completes the "partnership" vision. Not just observe and edit spatial structure, but observe and edit temporal structure.

---

## Node 10: The Quiet Revolution

If temporal tiles work, we've shown:

> Intelligence does not require continuous token-to-token mixing.
> It can emerge from discrete, addressable, stateful control loops.

That's not a tweak to transformers. That's a different answer to the question "how does sequence understanding work?"

And the answer is: **phases and transitions, not attention and mixing.**

**Implication**: This is either wrong (and we learn why) or revolutionary (and we have a new substrate). Either way, it's worth testing.

---

## Summary of Nodes

1. **State as contracted time** - not full history, just mode
2. **Soft states, hard transitions** - the interpretable hybrid
3. **Carry flag as proof** - 6502 already told us
4. **Bracket matching as atom** - simplest temporal test
5. **Temporal dispatch tables** - FSM extraction
6. **Attention replacement** - four roles, four solutions
7. **Long-range question** - the honest unknown
8. **Cognitive glue** - Levin's local-rule emergence
9. **Surgery through time** - complete the partnership
10. **Quiet revolution** - different substrate, not optimization

---

*Nodes extracted. Ready for convergence.*
