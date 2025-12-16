# Mesa 2: FFNv1 vs FFNv2 Delta - Raw Exploration
*New understanding: TriX is the cartographer, not the compute.*

---

## The Reframe

I was measuring the wrong thing.

I kept asking: "Which FFN computes more accurately?"
The real question: "Which FFN discovers and organizes better?"

TriX tiles aren't supposed to BE organs. They're supposed to FIND where organs should be. The 92% purity proves they can. The 100% accuracy is the organ's job, not TriX's.

So what does this mean for v1 vs v2?

---

## v1: The Discoverer

SparseLookupFFN v1 does one thing: route inputs to tiles based on learned signatures.

What it gives us:
- Routing decisions (which tile?)
- Tile outputs (what computation?)
- Balance loss (don't collapse to one tile)

What it DOESN'T give us:
- Why did it route there?
- What does each tile "mean"?
- Can we influence the routing?

v1 is a black box discoverer. It finds structure, but we can't see it or control it.

---

## v2: The Articulate Discoverer

SparseLookupFFNv2 adds:
- Claim tracking (which classes → which tiles)
- Surgery (insert/freeze/unfreeze signatures)
- Regularizers (ternary, sparsity, diversity)
- Score calibration (stable routing)

What this means in the new frame:

**Claim tracking** = We can SEE the map TriX draws
**Surgery** = We can EDIT the map
**Regularizers** = We can make the map CLEANER
**Calibration** = We can make the map STABLE

v2 isn't a better computer. It's a better CARTOGRAPHER with tools for collaboration.

---

## The Delta

v1: Discovers structure (black box)
v2: Discovers structure (transparent, editable, clean)

The compute is the same. The RELATIONSHIP with the structure is different.

v1 is like an explorer who finds a continent but can't explain where things are.
v2 is like an explorer who finds a continent AND draws a legible map AND lets you add annotations.

---

## New Thought: Organs as Tile Specializations

If TriX discovers where organs should be, can we MAKE tiles into organs?

Idea: After TriX discovers "Tile 4 handles ALU", we could:
1. Extract Tile 4's learned weights
2. Replace them with a proper ALU organ (from FLYNNCONCEIVABLE)
3. Freeze that tile
4. Now Tile 4 IS an ALU, not "kinda like an ALU"

This is SURGERY at a deeper level. Not just editing signatures, but TRANSPLANTING organs.

v2 enables this. v1 doesn't give us the hooks.

---

## New Thought: Routing as Instruction Decode

CPUs have instruction decoders. Given an opcode, route to the right execution unit.

TriX's routing IS instruction decode:
- Input: embedded opcode + operands
- Routing: signature matching → tile selection
- Output: execution by selected tile

v1 routing: learned, implicit, fixed after training
v2 routing: learned, observable, editable after training

v2 lets us SEE the instruction decoder's logic (claim tracking) and MODIFY it (surgery).

---

## New Thought: Composition via Multi-Layer Routing

The 6502 insight: organs compose into programs.

Multi-layer TriX: routing paths compose into... what?

Layer 1: Coarse category (ALU vs LOGIC vs SHIFT)
Layer 2: Fine operation (ADC vs SBC within ALU)
Layer 3: Specific case (overflow vs no-overflow within ADC)

The PATH through layers is like a decision tree. Or like microcode. Each layer refines the routing.

v2 claim tracking could show us these paths:
- "ADC with overflow goes: L1→Tile2, L2→Tile5, L3→Tile11"
- "ADC without overflow goes: L1→Tile2, L2→Tile5, L3→Tile8"

This is interpretable microcode. v1 has the paths but we can't see them.

---

## New Thought: The Organ Library

If FLYNNCONCEIVABLE gives us proven organs (100% each), and TriX discovers where they go, then:

1. Train TriX on mixed data → discovers organ boundaries (92% purity)
2. For each discovered region, identify what organ it needs
3. TRANSPLANT the proven organ into that tile (surgery)
4. Freeze transplanted organs
5. Now: TriX routing + proven organs = 100% accuracy

This is HYBRID architecture:
- Routing: learned (TriX)
- Compute: engineered (FLYNNCONCEIVABLE organs)

v2 makes this possible. v1 doesn't.

---

## New Thought: Factory Pattern

The "CPU factory" idea means:

FLYNNCONCEIVABLE organs = parts library
TriX = assembly instructions
Composition = final product

Different compositions = different CPUs:
- Standard 6502: organs wired one way
- Parallel 6502: multiple ALUs, TriX routes based on availability
- Extended 6502: add new organs, TriX learns to route to them

v2's surgery is the ASSEMBLY TOOL. You can:
- Insert new organs (new signatures)
- Wire them differently (freeze patterns)
- Extend capabilities (add tiles)

v1 gives you a fixed product. v2 gives you a factory.

---

## The Core Delta

v1: TriX as trained artifact
v2: TriX as living system

v1 discovers and freezes. Done.
v2 discovers, shows you, lets you edit, keeps working.

The delta isn't accuracy. It's AGENCY.

With v1, you train and hope.
With v2, you train, observe, intervene, refine, compose.

---

## What v2 Enables That v1 Doesn't

1. **Post-hoc interpretability**: After training, see what each tile learned
2. **Surgical correction**: If a tile learned wrong, fix it without retraining
3. **Organ transplant**: Replace learned approximations with proven implementations
4. **Compositional design**: Build processors by assembling organ-tiles
5. **Progressive refinement**: Start rough, observe, improve iteratively

None of these are possible with v1. You train and you get what you get.

---

## The Meta-Insight

v1 vs v2 isn't about the FFN. It's about the WORKFLOW.

v1 workflow: Design → Train → Deploy → Hope
v2 workflow: Design → Train → Observe → Understand → Edit → Refine → Deploy → Monitor → Improve

v2 is a PARTNERSHIP between human and model. v1 is a handoff.

---

## Fragments

- Could we visualize tile signatures as "organ fingerprints"?
- Surgery + claim tracking = debugger for semantic geometry
- The 92% purity is TriX saying "here's where I think the organs are"
- The 8% impurity is where TriX is uncertain - those are the INTERESTING cases
- What if we only let humans decide routing for the uncertain 8%? Human-in-the-loop routing.

---

*End raw dump. Let me see what's actually here.*
