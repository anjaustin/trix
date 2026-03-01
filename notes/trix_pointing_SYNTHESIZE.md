# SYNTHESIZE - What TriX Is Pointing At

Date: 2026-03-01
Method: Lincoln Manifold (Phase 4)
Goal: turn the reflection into a concrete direction that can guide future work.

## The Thesis (One Sentence)

TriX is an address-plane runtime for learned computation: it discovers, stabilizes, compiles, ships, and monitors discrete internal subroutines.

## The Product Shape

```
        +-----------------------------+
        |         CARTRIDGES          |
        |  (tiles / atoms / FFT / ..) |
        +--------------+--------------+
                       |
                       v
        +-----------------------------+
        |      ADDRESS PLANE OS       |
        | route / telemetry / bundle  |
        | compile / validate / guard  |
        +--------------+--------------+
                       |
                       v
        +-----------------------------+
        |         ACCELERATION        |
        | native kernels (boring)     |
        +-----------------------------+
```

The OS layer is the real differentiator. Cartridges can evolve; the OS layer must be stable.

## Concrete Commitments (If We Stay Honest)

1) All claims must be typed
- guarantee: invariant enforced by tests
- condition: guarantee holds only under documented constraints
- falsified: counterexample exists and is tested
- aspiration: hypothesis not yet validated

2) All routing telemetry must declare semantics
- backend: hierarchical_dot vs flat_popcount
- score domain: score-space vs gate-space
- margin scope: cluster-local vs global

3) Bundles are versioned firmware
- compatibility rules are documented
- loading performs validation and emits a report

## The Clean Next Step (Beyond Current Mesa)

Mesa 15 proposal: Address Integrity

Deliverables:
- Bundle signing/checksums (not crypto-heavy at first; just integrity and reproducibility)
- Address drift policies:
  - what constitutes invalidation?
  - how do we recompile safely?
- A minimal "ABI" for tiles:
  - input/output shape
  - expected invariants
- A policy mechanism:
  - allow/deny lists of addresses
  - fallback behavior specification

Why this is the grain:
- once you can treat addresses as stable and enforceable, TriX becomes infrastructure.

## Success Criteria (Developer Experience)

A first-time developer should be able to:

1) Install and trust:
- run `trix doctor` and see PASS

2) See behavior:
- run `trix bench` and get artifacts

3) Ship behavior:
- export a bundle, load it elsewhere, validate it

4) Understand limits:
- read `docs/KNOWN_LIMITS.md` and see counterexamples as intentional boundaries

## The Fun Part (Why This Is Worth It)

If this works, we get a new way to build ML systems:
- small subroutines that can be addressed
- contracts that can be compiled
- bundles that can be shipped
- drift that can be detected

That is closer to software engineering than to training rituals.
