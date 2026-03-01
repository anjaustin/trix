# Lifecycle v1 Notes

This note documents what the v1 lifecycle wrapper provides, and what it does *not* guarantee.

File: `src/trix/nn/lifecycle.py`

## Surface

- `observe(...)`: run forward and optionally emit JSONL telemetry
- `edit_insert_signature(...)` + `undo(...)`: reversible surgery wrapper
- `compile_stable(...)` + `export_dispatch_table(...)` + `import_dispatch_table(...)`
- `monitor()`: consolidated stats
- `stability_probe(...)`: controlled signature-perturbation churn probe

## Telemetry Semantics (Important)

### Hierarchical Routing

`SparseLookupFFNv2.route(...)` performs hierarchical routing:

1) choose a cluster using cluster signatures
2) choose a tile within the selected cluster

When `return_scores=True`, the returned `all_scores` tensor is populated only for the tiles in the selected cluster for each input. Other columns are left at 0.

Implication:
- Any tie/margin stats computed from `all_scores` are *cluster-local*, not a full "global over all tiles" margin.
- Cluster-level tie degeneracy can exist even when tile-level margins are large.

### Score Calibration

When enabled, routing decisions use `ScoreCalibrationSpline` outputs, not raw dot-product scores.

The spline knot values are learnable and are not constrained to be monotone. In a non-monotone regime:

- `argmax(score)` can differ from `argmax(calibrator(score))`

If you log margins, you must be explicit about whether the margin is in score-space or gate-space.

## Falsification Tests

We include explicit falsification tests for these limitations:

- `tests/test_lifecycle_falsify.py`

These tests assert that counterexamples exist; they are not treated as failures.
