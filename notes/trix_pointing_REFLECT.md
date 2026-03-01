# REFLECT - What TriX Is Pointing At

Date: 2026-03-01
Method: Lincoln Manifold (Phase 3)
Inputs: `notes/trix_pointing_RAW.md`, `notes/trix_pointing_NODES.md`

## Core Insight

TriX is pointing at an "address plane" for learned computation.

Not "a model".

An address plane is a discrete internal namespace that selects subroutines. If it is stable, then:
- debugging becomes possible (why did tile 17 fire?)
- patching becomes possible (change tile 17 signature)
- deployment becomes possible (compile a contract)
- auditing becomes possible (which addresses are allowed?)

This is an OS-like framing: the compute is modular, the selection is observable, and the behavior is portable via bundles.

## The Structure Beneath The Repo

The repo contains several strands that look different but share a skeleton:

1) Discover: induce tiles (subroutines) + signatures (addresses)
2) Observe: measure distribution health, ties, margins, churn
3) Edit: surgery on signatures, enforce constraints
4) Compile: contract stable regions/classes into deterministic dispatch
5) Validate: native-vs-reference, compiled-vs-forced equivalence
6) Ship: bundle the address plane + contract
7) Monitor: drift detection, invalidation rules

The "novel" part is the lifecycle, not any single FFN variant.

## What Makes This Different From Standard MoE

Standard MoE: routing is typically learned, parameterized, and mostly opaque.

TriX direction: routing is treated as an artifact with:
- discrete representations (ternary signatures, packed distances)
- invariants and counterexamples
- compile/guard/monitor semantics

The OS comparison holds because the artifacts are explicit and portable.

## The Grain: Where The Wood Wants To Split

The system wants to split into two layers:

Layer A: Address Plane Runtime
- routing primitives
- telemetry and drift
- bundles and validation
- contracts

Layer B: Cartridges / Compute Libraries
- FFN variants
- transforms
- circuit atoms
- task-specific tiles

If we keep mixing them, the repo feels like a manifesto.
If we split them cleanly, it feels like infrastructure.

## The Delta: Boundary Cases That Matter

Per LMM, the deltas are where mistakes hide. For TriX, the deltas are:

1) Routing equivalence boundaries
- dot vs popcount is not universal; zeros and masking matter
- continuous magnitude information matters

2) Tie/near-tie regimes
- collapse and churn are not bugs; they are geometry
- tie-breaking must be explicit

3) Hierarchical semantics
- cluster-local margins are not global margins
- telemetry must declare what it is measuring

4) Approximation knobs (alpha scales)
- semantically correct for the declared computation
- not a universal approximation guarantee

These deltas are not embarrassing; they are the foundation of trust.

## The Future-True Thesis

If TriX succeeds, it will be because it makes a new interface possible:

"I can ship internal subroutines as addresses, and I can reason about what runs."

This is a compute story:
- addresses become an ABI
- bundles become deployable firmware
- contracts become policies
- routing becomes the instruction decoder

And it is also a safety story:
- certify allowed address paths
- detect drift of address semantics
- revoke bundles

## What I Now Believe

- The most important work is making address semantics stable and legible.
- The ternary substrate is powerful because it makes addressing discrete and cheap.
- The repo should lean into being infrastructure (runtime + toolchain) rather than a single model.

The clean next Mesa is not "more routing". It is "address plane maturity": versioning, compatibility, audit, policy.
