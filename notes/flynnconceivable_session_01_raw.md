# FLYNNCONCEIVABLE × TriX: Raw Exploration
*Stream of consciousness. No editing. See what emerges.*

---

## First Contact

A neural network that IS a CPU. Not simulates. IS.

That word choice matters. "Simulates" implies a gap between the model and the thing. "IS" collapses that gap. The weights ARE the logic gates. The inference IS the computation.

This is the same move as "routing IS the computation" in TriX. Not routing THEN computation. The act of selecting is the act of computing.

---

## The 6502 as Forcing Function

Why the 6502? Why not something simpler, or something modern?

The 6502 is:
- Small enough to be exhaustively testable
- Complex enough to have real structure (carry chains, overflow detection, flag interactions)
- Discrete - no approximation allowed
- Historically significant - it powered the Apple II, the NES, the C64

But I think the real reason is: the 6502's operations are SEMANTICALLY DISTINCT. ADC is not "kind of like" ORA. They're fundamentally different operations that happen to share an 8-bit bus.

This is the perfect test for semantic geometry. If TriX can learn 6502 operations, the tiles MUST specialize. There's no continuous interpolation between addition and XOR.

---

## Soroban Encoding

This stopped me. Thermometer encoding for arithmetic.

Value 37:
```
Rod 0 (1s):   ●●●●●●●○  = 7
Rod 1 (10s):  ●●●○○○○○  = 3
```

Why does this work? Because carry propagation is LOCAL in this representation. When you add two soroban numbers, the carry ripples visibly through adjacent rods. The network can SEE the structure.

Binary encoding hides this. 37 = 00100101. The relationship between bits isn't spatially apparent.

This is a lesson: REPRESENTATION DETERMINES LEARNABILITY. The same information, encoded differently, can be trivial or impossible to learn.

TriX implication: Are our signatures the right representation? What would a "soroban encoding" for language look like?

---

## The Spline Revelation

3.7MB → 3KB. Same accuracy.

This isn't compression in the information-theoretic sense. The MLP has millions of parameters, but almost all of them are encoding structure that could be expressed more directly.

The spline version says: "The function is piecewise linear. Here are the pieces."

That's what TriX is doing. The tiles are the pieces. The routing finds which piece you're in. The computation is linear within each piece.

The 6502's operations are piecewise linear in the right representation. ADC with soroban encoding is literally piecewise linear - the slopes encode the carry behavior at each grid cell.

---

## What Are The Organs?

FLYNNCONCEIVABLE has:
- ALU (ADC, SBC) - arithmetic with carry
- LOGIC (AND, ORA, EOR, BIT) - bitwise operations
- SHIFT (ASL, LSR, ROL, ROR) - bit manipulation with carry
- INCDEC (INC, DEC, INX, DEX, INY, DEY) - ±1 operations
- COMPARE (CMP, CPX, CPY) - subtraction for flags only
- BRANCH (BEQ, BNE, BCS, BCC, BMI, BPL, BVS, BVC) - flag testing

These are natural categories. A human designer would group them the same way.

But the neural network wasn't told about these categories. The MLP just learned input→output mappings. Yet the spline version recovers the structure.

What if TriX could discover these categories from the data alone? Route arithmetic to one tile, logic to another, shifts to a third - without supervision?

---

## The Surgery Connection

FLYNNCONCEIVABLE's spline organs are hand-designed. Someone figured out the grid structure, the coefficients, the piecewise linear approximations.

But what if you could LEARN the spline structure? Start with a guess, let gradient descent refine it?

That's exactly what TriX v2's surgery API enables:
1. Insert a designed signature (hypothesis about semantic region)
2. Freeze it (force the system to organize around it)
3. Train (let gradient descent fill in the rest)
4. Unfreeze (let the system refine your hypothesis)
5. Observe (did it keep your structure or find something better?)

We could INSERT signatures for "arithmetic" and "logic" and see if the system builds organs around them.

---

## The Composition Question

A CPU isn't just organs. It's organs COMPOSED. A program is a sequence of operations, each feeding into the next.

```
LDA #$25    ; Load 37
CLC         ; Clear carry
ADC #$1A    ; Add 26 → 63
ASL A       ; Shift left → 126
```

This is a PATH through organs: LOAD → FLAG → ALU → SHIFT

TriX's multi-layer routing creates paths. Layer 1 routes to coarse regions, Layer 2 refines, Layer 3 specializes.

Could a 3-layer TriX learn that "arithmetic followed by shift" is a common pattern and create a dedicated path for it?

---

## The Verification Obsession

460,928 combinations. Zero errors.

This is unusual for neural networks. We usually accept 99.x% accuracy. But FLYNNCONCEIVABLE demands 100% because CPUs don't get to be approximate.

What does it take to get 100%?
- Exhaustive training data (every combination)
- Oversampling edge cases (zeros, wraparounds)
- Architectural choices that match the problem (soroban for ALU)
- Dedicated heads for tricky outputs (Z flag gets extra weight)

TriX for 6502 would need the same rigor. No "usually works." Perfect or nothing.

---

## What Scares Me

The MLP achieves 100% accuracy with brute force. Millions of parameters memorizing the input-output mapping.

The spline achieves 100% with structure. 3KB of coefficients encoding the actual logic.

What if TriX finds a THIRD way? Neither memorization nor explicit structure, but something emergent?

What if the tiles discover patterns we don't have names for? Groupings that aren't ALU/LOGIC/SHIFT but something more fundamental?

That's exciting and terrifying. We might learn something about computation itself.

---

## The Real Question

FLYNNCONCEIVABLE proves neural networks can learn exact discrete computation.

TriX claims routing IS addressing IS computation.

The synthesis: Can TriX learn to BE a 6502 by discovering that CPU operations are geometric regions in signature space?

If yes: We've shown that semantic geometry isn't just a metaphor. It's how discrete computation can be encoded in continuous space.

If no: We learn where the thesis breaks down. Also valuable.

---

## Fragments

- The 6502 is a formal system. Complete, consistent, decidable. Can TriX learn formal systems?
- Branch prediction is routing. Modern CPUs route instructions to execution units. We're not far from real architecture.
- What's the "vocabulary" of 6502 operations? 56 opcodes, ~150 with addressing modes. Smaller than most language models.
- The carry flag threads through everything. It's the hidden state. TriX would need to learn this implicit dependency.
- Error correction: What if a tile gets an operation wrong? CPUs have no tolerance. Would TriX need redundancy?

---

*End raw dump. Time to see what's actually here.*
