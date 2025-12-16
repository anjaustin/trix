# Temporal Tiles: Raw Exploration
*Stream of consciousness. No editing. Let it flow.*

---

## The gap that's been nagging

We built spatial routing. Tiles specialize to WHAT. Class 0 goes to tile 2. AND operations go to LOGIC tiles. The cartographer finds regions.

But there's something attention does that we haven't touched: **binding across time**.

Not "which token attends to which" - that's the implementation. The FUNCTION is: the model knows that something from 50 tokens ago matters NOW. It carries relevance forward. It maintains context.

TriX routing is stateless. Each token routes independently. The tile doesn't know what came before. It can't count. It can't track. It can't remember.

---

## What is state, really?

State is... contracted time. It's the past compressed into something the present can use.

Attention does this by keeping all past tokens accessible and softly mixing them. Expensive. Beautiful. But is it necessary?

What if instead of "which past tokens matter?" you asked "which REGIME am I in?"

Regime = a mode of operation. A phase. A context.

- "I'm inside a string literal"
- "I'm in the middle of a multi-digit number"
- "I'm processing the THEN branch of an IF"
- "The previous operation set the carry flag"

These aren't token references. They're STATE LABELS. Discrete. Finite. Nameable.

---

## The FSM intuition

A finite state machine has:
- States (finite, discrete)
- Transitions (input × state → new state)
- Outputs (state → output)

Classical FSMs are designed by hand. You draw the state diagram. You enumerate the transitions.

What if you LEARNED the FSM?

- States: a continuous latent vector (soft state)
- Transitions: temporal tiles (discrete selection, continuous update)
- The state space emerges from training

Soft states, hard transitions. That's the hybrid.

---

## Why hard transitions matter

Soft state transitions (like RNNs) have a problem: everything blurs. The hidden state becomes an uninterpretable soup. You can't point to it and say "this is the counting state."

Hard transitions (tile selection) create BOUNDARIES. Discrete events. Observable moments where the system says "I'm switching modes now."

That's interpretability through time, not just space.

---

## The carry flag haunting

ADC at 4.8% accuracy. Everything else at 95%+. Why?

Because ADC(a, b, c) depends on the carry flag from the PREVIOUS operation. It's the only operation where the past matters.

We hacked it with soroban encoding - putting the carry in the input representation. But that's cheating. We told the model what it should have learned.

A temporal tile could learn: "After an operation that overflows, I enter the carry-set regime. In this regime, ADC adds one more."

That's not input encoding. That's STATE.

---

## What would routing look like?

Spatial: `input → signature match → tile`

Temporal: `(input, state) → signature match → tile → (output, new_state)`

The signature now has two parts:
- What input activates me?
- What state activates me?

Or maybe it's simpler: concatenate input and state, match against temporal signatures.

```python
combined = concat(x_t, state_{t-1})
scores = combined @ temporal_signatures.T
tile_id = argmax(scores)
```

Same routing logic. Extended domain.

---

## State dimensionality

How big is state?

Too small: can't represent enough regimes. Can't count high enough. Can't track nested structures.

Too big: just reinventing hidden states. Becomes a soup. Loses interpretability.

Sweet spot: small enough to be interpretable, big enough to be useful.

Maybe 16-64 dimensions? Enough for a few independent "flags" or "counters" but not a full representation.

Or maybe state should be STRUCTURED:
- 4 bits for "depth" (0-15)
- 4 bits for "mode" (16 possible modes)
- 8 bits for "context tag"

Structured state = more interpretable. But less learnable?

---

## The bracket matching test

Simplest temporal task: count parentheses.

```
( ( ) ) → valid (depth: 0→1→2→1→0)
( ) ) ( → invalid (depth goes negative)
```

Flat model can't do this. It sees each paren independently. No memory.

Attention can do this by attending to all previous tokens and somehow computing depth. Expensive and indirect.

Temporal tiles could do this with ~4 tiles:
- Tile 0: "at depth 0" - ( increments, ) rejects
- Tile 1: "at depth 1" - ( increments, ) decrements
- Tile 2: "at depth 2" - same pattern
- Tile 3: "overflow" - rejects everything

Four tiles. Four states. A learned counter.

If this works, we've shown discrete state routing can replace attention for algorithmic tasks.

---

## The compilation question

If temporal tiles work, can we compile them?

Spatial compilation: `class → tile`

Temporal compilation: `(state_regime, input_class) → (tile, next_regime)`

That's a state transition table. Literally the definition of an FSM.

```
current_state | input | next_state | tile
-------------|-------|------------|-----
DEPTH_0      | (     | DEPTH_1    | tile_1
DEPTH_0      | )     | ERROR      | tile_err
DEPTH_1      | (     | DEPTH_2    | tile_2
DEPTH_1      | )     | DEPTH_0    | tile_0
...
```

That's not a neural network. That's a PROGRAM. Extracted from learning. Serializable. Verifiable. Deployable.

---

## The Michael Levin connection

Levin talks about "cognitive glue" - how cells coordinate to form organs, how organs coordinate to form organisms. Not through central control, but through local rules and state.

Each cell has state. Each cell responds to neighbors. Patterns emerge.

Temporal tiles are cognitive glue for sequences:
- Each position has state (inherited from previous)
- Each position routes based on (input, state)
- Patterns emerge (syntax, semantics, structure)

Not attention (global coordination). Not RNN (smooth flow). Discrete local rules with emergent global behavior.

---

## What scares me

Long-range dependencies. Attention can bind token 1 to token 1000 directly. State has to CARRY that information through 999 intermediate steps.

Can discrete state survive that? Or does it forget?

Maybe the answer is: some things need attention, some things need state routing. Hybrid architectures. Use the right tool for the right dependency.

Or maybe: if you have enough state dimensions and the right tile structure, long-range emerges from local rules. Like how cellular automata can transmit signals.

I don't know. That's what experiments are for.

---

## What excites me

The interpretability. 

Imagine looking at a trained temporal tile system and being able to say:
- "Tile 3 is the 'inside string literal' state"
- "Tile 7 is the 'accumulating digits' state"
- "The transition from Tile 3 to Tile 0 happens on closing quote"

That's not post-hoc interpretation. That's READING THE PROGRAM.

And then you can EDIT it:
- "Tile 3 should also activate on backtick for template literals"
- Insert the transition. Freeze it. Done.

Surgery through time, not just space.

---

## The quiet revolution

If this works:

Attention is not necessary for sequence modeling.
Attention is one implementation of temporal binding.
Discrete state routing is another.
And the discrete one is interpretable, compilable, editable.

That's not "attention-free transformers."
That's a different computational model.
One where the program is readable.
One where intelligence is engineered, not just trained.

---

## What wants to emerge

A temporal tile layer that:
1. Maintains explicit state (small, structured)
2. Routes based on (input, state)
3. Updates state discretely (tile selection is the transition)
4. Can be compiled to a state machine
5. Can be surgically edited

And a test:
- Bracket matching (counting)
- Then: arithmetic with carry
- Then: simple parsing (if/then/else)
- Then: maybe language modeling?

Start small. Prove the atom. Then build molecules.

---

*End raw exploration. Let it sit.*
