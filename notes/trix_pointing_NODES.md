# NODES - What TriX Is Pointing At

Date: 2026-03-01
Method: Lincoln Manifold (Phase 2)
Source: `notes/trix_pointing_RAW.md`

## Node 1: TriX Is Becoming A Routing OS
Observation: features cluster around lifecycle (observe/edit/compile/bundle/monitor), not around a single FFN architecture.
Why it matters: this reframes success from "better FFN" to "addressable compute runtime."

## Node 2: Address Plane Is The Product Surface
Observation: signatures + routing are the developer-facing abstraction.
Why it matters: stability of address semantics is more important than raw model capacity.

## Node 3: Contracts Are The Deployable Unit
Observation: compiled dispatch and bundles are contracts: "for region/class X, run tile Y." Guardrails + fallbacks define correctness.
Why it matters: contractability is what turns research into deployable systems.

## Node 4: Observability Is The Real Differentiator
Observation: tie/near-tie, margins, churn curves, drifted classes, hit rates.
Why it matters: conditional compute without observability is unshippable.

## Node 5: Falsification Is A Feature, Not A Bug
Observation: explicit counterexamples exist and are documented.
Why it matters: skeptics trust boundaries; teams can build guardrails.

## Node 6: Ternary Enables A Discrete Address ABI
Observation: ternary packing + XOR+POPCNT turns routing into a cheap discrete primitive.
Why it matters: makes "addressing" feel like compute infrastructure rather than learned soup.

## Node 7: Multiple Routing Backends = A Governance Problem
Observation: hierarchical dot vs flat popcount have different semantics.
Why it matters: if the user doesn't know which backend they're using, telemetry becomes misleading.
Tension: flexibility vs simplicity.

## Node 8: Stability Is The Central Hard Problem
Observation: addresses must remain meaningful across training drift and distribution shift.
Why it matters: without stability, addressability is a mirage.

## Node 9: Cartridges Are Attempts At Fixed Semantics
Observation: FFT/WHT/adder/opcodes are attempts to load deterministic compute into the routing substrate.
Why it matters: suggests a long-term direction: verified tile libraries.

## Node 10: Compute-As-Routing Is Piecewise Function Machines
Observation: partition input space into regions; each region invokes a subroutine.
Why it matters: connects TriX to program synthesis, decision trees, mixture models, and hardware dispatch.

## Node 11: Memory + Sequencing Are The Missing Atoms
Observation: universal computation needs state and iteration, not just routing.
Why it matters: determines whether TriX is a feedforward "selector" or a real machine.

## Node 12: Security/Safety Becomes Address Integrity
Observation: if addresses are stable, they can be audited, signed, certified.
Why it matters: a new safety interface: "what internal subroutines are allowed to run."

## Node 13: Alpha Scales Are A Separate Dimension
Observation: BitNet-style ternary+alpha is semantically correct but not a universal approximation.
Why it matters: performance/approximation knobs must be framed as bounded, measurable tradeoffs.

## Node 14: Developer Delight Requires A Narrow Waist
Observation: many modules exist; only one path should be blessed.
Why it matters: first-time experience is the adoption bottleneck.
