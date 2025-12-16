# Mesa 2: FFNv1 vs FFNv2 Delta - Reflection
*Finding the nodes. Following the threads.*

---

## The Nodes

Reading back, these ideas have weight:

1. **Cartographer, not compute** - TriX's role is discovery
2. **Transparent vs opaque** - v2 shows what v1 hides
3. **Organ transplant** - Surgery enables hybrid architectures
4. **Routing as instruction decode** - The mapping to CPU architecture
5. **Factory vs artifact** - v2 is generative, v1 is fixed
6. **Agency** - The human-model relationship changes

---

## Node 1: Cartographer, Not Compute

This reframe changes everything.

If TriX's job is to FIND structure, not BE structure, then measuring accuracy is wrong. We should measure:
- Purity (how clean are the discovered regions?)
- Stability (do regions stay consistent?)
- Interpretability (can we understand what was discovered?)

v1 can find structure (92% purity proves it).
v2 can find structure AND show us what it found.

The compute accuracy depends on what we PUT in the regions after discovery. If we put approximate learned functions, we get approximate accuracy. If we transplant proven organs, we get 100%.

**v2's advantage: It lets us SEE the discovery and ACT on it.**

---

## Node 2: Transparent vs Opaque

v1 is a trained artifact. Query it, get answers, don't ask why.

v2 is a trained artifact WITH ANNOTATIONS:
- Claim matrix: "Tile 4 saw 55% ADC, 30% SBC, 15% other"
- Signature analysis: "Tile 4 uses dims [3,7,12] positive, [8,9] negative"
- Surgery history: "Tile 4 was edited at step 1000"

This transparency is valuable even if compute accuracy is identical because:
- Debugging: When something fails, we can see WHY
- Trust: We can verify the model's organization makes sense
- Iteration: We can improve specific tiles without retraining everything

**v2's advantage: Observability.**

---

## Node 3: Organ Transplant

This is the big idea.

TriX discovers: "Region X should handle arithmetic"
FLYNNCONCEIVABLE provides: Proven ALU organ (100% accurate)
Surgery enables: Transplant ALU into Region X

The result: TriX's discovered structure + proven compute = best of both.

This is impossible with v1. There's no surgery API, no way to replace tile internals.

v2's architecture was designed for this even if we didn't realize it:
- `insert_signature()` = define what the tile responds to
- `freeze_signature()` = lock the routing
- Tile weights can be loaded from pretrained organ

**v2's advantage: Hybrid architecture is possible.**

---

## Node 4: Routing as Instruction Decode

Modern CPUs separate:
- Instruction decode (what operation? which unit?)
- Execution (do the operation)

TriX separates:
- Routing (which tile?)
- Tile compute (what output?)

This isn't a metaphor. It's the same architecture pattern.

v1 routing: Learned lookup table (implicit)
v2 routing: Learned lookup table (explicit, editable)

The instruction decoder in a CPU is designed. In TriX, it's learned. But v2 lets us REFINE the learned decoder with design knowledge.

**v2's advantage: The decoder is editable.**

---

## Node 5: Factory vs Artifact

v1 produces a trained model. Done. Ship it.

v2 produces a trained model PLUS:
- A map of what it learned (claim tracking)
- Tools to modify it (surgery)
- Quality controls (regularizers)

This means v2 isn't just a model, it's a MODEL FACTORY. You can:
- Start with baseline training
- Observe what emerged
- Edit to fix problems
- Add new capabilities
- Ship version N
- Observe in production
- Edit to fix new problems
- Ship version N+1

v1 requires retraining from scratch for each version.
v2 allows iterative refinement.

**v2's advantage: It's a living system, not a frozen artifact.**

---

## Node 6: Agency

This is the meta-level.

v1: Model does what it learned. Human accepts result.
v2: Model does what it learned. Human sees what it learned. Human can change it. Model adapts.

The relationship shifts from:
- Master/servant (human commands, model obeys training)
to:
- Collaboration (human and model co-create the organization)

Surgery isn't just an API. It's a DIALOGUE:
- Model: "I think ADC goes in Tile 4"
- Human: "Actually, Tile 4 should also handle SBC. Let me adjust the signature."
- Model: "Okay, I'll route SBC there too now."

**v2's advantage: Human-model collaboration.**

---

## The Synthesis

The delta between v1 and v2 isn't about which computes better.

It's about which ENABLES more:

| Capability | v1 | v2 |
|------------|----|----|
| Discover structure | ✓ | ✓ |
| See discovered structure | ✗ | ✓ |
| Edit discovered structure | ✗ | ✓ |
| Transplant proven compute | ✗ | ✓ |
| Iterate without full retrain | ✗ | ✓ |
| Collaborate with human | ✗ | ✓ |

v1 is a TOOL.
v2 is a WORKSHOP.

---

## What Emerged

The 6502 mesa taught us: TriX finds structure, organs fill it.

The v1/v2 mesa teaches us: v1 finds structure, v2 finds structure AND lets us work with it.

The combination: 

1. Train TriX v2 on mixed data
2. v2 discovers organ boundaries (claim tracking shows us)
3. For unclear boundaries, human intervenes (surgery)
4. For proven regions, transplant organs (surgery)
5. Result: Human-guided, organ-powered, TriX-routed system

This isn't v1 vs v2. This is v2 ENABLING A NEW PARADIGM.

---

## The Question That Remains

If v2 enables organ transplant, and FLYNNCONCEIVABLE organs are proven at 100%, then:

Can we build a TriX-routed 6502 that achieves 100% by transplanting organs into discovered regions?

This would be:
- Routing: Learned by TriX (discovers structure)
- Compute: Engineered by FLYNNCONCEIVABLE (100% proven)
- Assembly: Enabled by v2 surgery (hybrid architecture)

The ultimate test isn't v1 vs v2 accuracy. It's: Can v2 enable something neither could do alone?

---

*Time to find convergence.*
