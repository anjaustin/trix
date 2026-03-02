# Future Development Plan

This doc captures the most reachable TriX use cases and the concrete next steps to make them delightful.

Note: this repo is archived; active development lives in `../trix-z/`.

## Reachable Use Cases (1..5)

1) Drop-in FFN replacement (vanilla PyTorch Transformer)
- Default path: `trix.nn.DropInFFN` in `dynamic` mode.
- Goal: keep `forward(x)->y` signature and require no training framework changes.
- Success: a user swaps one module and can run `trix doctor` + `trix bench` to validate.

2) Contracted inference for known classes / stable partitions
- Path: profile with labels -> `compile_stable` -> deploy with `class_hint` + confidence + guardrails.
- Goal: deterministic dispatch when hints are available; explicit fallback when not.
- Success: contract hit-rate and fallback reasons are visible in telemetry.

3) Routing stability research loop
- Path: `trix bench` drift benchmark + lifecycle telemetry (ties/near-ties/margins/churn).
- Goal: make routing changes falsifiable and comparable.
- Success: stability regressions show up as changed churn curves and tie/near-tie rates.

4) Portable address-plane deployment
- Path: `trix export-bundle` / `trix load-bundle`.
- Goal: ship routing artifacts (compressed signatures + contract + validation) across machines/runs.
- Success: bundle validation passes and routing decisions are reproducible under declared backend semantics.

5) CPU-side packed ternary inference experiments
- Path: optional native kernel + strict reference-vs-native correctness harness.
- Goal: allow performance work without semantic drift.
- Success: native speedups are measurable, and tests prevent silent mismatches.

## Next Steps

## Next Mesa Of Innovation: Shippable Address Plane

The next Mesa that feels like a capability jump (not just features) is making the address plane *shippable*:

1) **Bless one routing backend as the reference**
- Pick a single backend + configuration as the semantic baseline.
  - Define tie-break rules, near-tie semantics, normalization, and what constitutes "equivalence".

2) **Make the Address ABI explicit and enforceable**
- A stable schema for routing events and stability events (versioned).
- A clear compatibility rule: what can change without invalidating a bundle/contract.

3) **Bundle becomes the deployable artifact**
- `export-bundle` emits: signatures (optionally compressed) + optional compiled dispatch + validation report + manifest.
- `load-bundle` enforces compatibility and runs a small semantic spot-check.

4) **Drift becomes runtime behavior, not just telemetry**
- A drift policy decides: use contract vs fall back to dynamic routing, with a reason code.
- Report hit/miss/fallback + reason as a first-class artifact.

### What Makes This A Mesa

You can take a trained router, freeze it into a contract, ship it, and know when it is safe to use it (and when to bail out) with auditability.

### How To Prove It (Cartridges)

- Cartridge A: "stable partition"
  - Demonstrate compiled dispatch matches dynamic routing for declared stable classes.
  - Output golden artifacts + fail loudly on mismatch.
- Cartridge B: "drift under continued training / shift"
  - Produce churn curve + contract hit-rate curve + margin/near-tie stats.
  - Establish falsification thresholds (regressions become test failures).

- Strengthen the "drop-in" story with a minimal PyTorch Transformer example wired to `DropInFFN`.
- Add a short guide: "dynamic -> contracted" migration (what changes, what stays same).
- Extend `trix bench` to print a one-line drift summary (max churn, mean churn, max drifted classes).
- Add a bundle compatibility policy (what v1 guarantees across versions) to `docs/BUNDLE_SCHEMA.md`.
