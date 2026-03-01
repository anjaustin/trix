# Convergent Ideas: What TriX Is Really About

This document captures the essential insights of the TriX project,
distilled from extended reflection. It's meant to clarify the core
contributions and guide future development.

---

## The Central Insight

**Constraint enables emergence.**

When weights are limited to {-1, 0, +1}, they become legible. Each
weight answers a simple question: "Do I want this feature (+1), its
opposite (-1), or do I not care (0)?"

This legibility has consequences:

1. **Routing emerges from structure.** The sum of a tile's weights
   creates a signature. The signature IS the address. No learned
   router needed.

2. **Compression is automatic.** Three values fit in 2 bits. Four
   weights pack into one byte. 16x memory reduction follows directly.

3. **Introspection becomes possible.** You can read a ternary weight
   and understand it. This opens the door to systems that observe
   their own structure.

The standard approach adds capacity to solve problems. TriX subtracts
capacity and finds that structure solves different problems.

---

## The Three Layers

### Layer 1: Kernel (Solid)

The foundation. 2-bit packing, NEON acceleration, fast inference.

- **Claim**: Ternary weights can be stored and computed efficiently.
- **Status**: Proven. The math is straightforward, the code works.
- **Audience**: Anyone deploying on constrained hardware.

### Layer 2: Architecture (Validated)

Sparse FFN with emergent routing. Hierarchical scaling.

- **Claim**: Signature-based routing works without learned gates.
- **Status**: Validated on benchmarks (14% perplexity improvement on
  TinyShakespeare with 2.3x fewer parameters).
- **Audience**: ML practitioners wanting sparse architectures.

### Layer 3: Meta-Learning (Experimental)

Self-observation during training. Intervention based on trajectory.

- **Claim**: Watching training dynamics enables better guidance than
  blind optimization.
- **Status**: Experimental. Interesting but unproven at scale.
- **Audience**: Researchers interested in training dynamics.

---

## What's Novel

1. **Zero-parameter routing.** Other sparse/MoE architectures learn
   routing. TriX derives routing from weight structure. This is
   architecturally simpler and more interpretable.

2. **Legibility by design.** Ternary weights aren't just compressed;
   they're readable. Each tile's signature has semantic meaning
   derived from its weights.

3. **Observation-based training** (experimental). The meta-learning
   layer treats training as something to be understood, not just
   executed.

---

## What's Not Novel (And That's Fine)

- Ternary quantization exists elsewhere (TTQ, TWN, etc.)
- Mixture of experts exists elsewhere
- Meta-learning exists elsewhere

The novelty is the combination and the specific insight about routing.
TriX doesn't need to pretend it invented quantization.

---

## The Accessibility Question

Different people need different things from this project:

### The Practitioner

"I want to compress my model and run it fast."

**Give them:**
- `TriXLinear` as a drop-in replacement for `nn.Linear`
- `TriXFFN` as a drop-in replacement for feedforward layers
- Clear benchmarks showing memory/speed tradeoffs
- Examples that work in 10 lines of code

**Don't burden them with:**
- Philosophy about emergence
- Experimental meta-learning
- Theory about addressing

### The Researcher

"I want to understand the contribution and extend it."

**Give them:**
- Clear problem statement
- Precise claims with evidence
- Reproducible experiments
- Honest discussion of limitations

**Don't alienate them with:**
- Spiritual language
- Grandiose framing
- Unsubstantiated claims

### The Learner

"I want to understand how this works."

**Give them:**
- Conceptual explanations before code
- Worked examples
- Intuition about why ternary creates signatures
- A path from simple to complex

**Don't overwhelm them with:**
- All the layers at once
- Jargon without definition
- Code without context

---

## Cleanup Priorities

1. **Separate stable from experimental.** The kernel and core NN
   layers should be clearly production-ready. The meta-learning layer
   should be clearly marked as research.

2. **Remove spiritual language.** "Guardian Angel" → "Training
   Observer." "Love as the process" → delete. "RLHF is dead" → delete.
   Keep the technical substance, lose the framing.

3. **Add benchmarks.** Comparisons to standard approaches. Clear
   numbers. Reproducible conditions.

4. **Write for three audiences.** A quickstart for practitioners. A
   paper-style document for researchers. A tutorial for learners.

5. **Keep what's real.** The core insight is genuine. The compression
   is real. The architecture works. Don't lose substance while
   removing noise.

---

## The Name

"TriX" is fine. It suggests ternary (tri-) and connects to the
matrices/tricks domain. Not too clever, not too bland.

"HALO" and "Guardian Angel" and "Mesa" should probably go or be
clearly relegated to historical/experimental context.

---

## Final Note

This project contains real ideas wrapped in presentation that
undermines them. The cleanup is about letting the ideas speak clearly.

The constraint-enables-emergence insight is worth preserving.
The self-observation-during-training idea is worth exploring.
The packaging is worth discarding.

What emerges should be clean, useful, and honest about what it knows
and what it's still figuring out.
