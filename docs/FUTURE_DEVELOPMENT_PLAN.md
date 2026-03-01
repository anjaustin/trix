# Future Development Plan

This doc captures the most reachable TriX use cases and the concrete next steps to make them delightful.

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

- Strengthen the "drop-in" story with a minimal PyTorch Transformer example wired to `DropInFFN`.
- Add a short guide: "dynamic -> contracted" migration (what changes, what stays same).
- Extend `trix bench` to print a one-line drift summary (max churn, mean churn, max drifted classes).
- Add a bundle compatibility policy (what v1 guarantees across versions) to `docs/BUNDLE_SCHEMA.md`.
