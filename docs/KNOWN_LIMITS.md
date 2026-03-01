# Known Limits and Counterexamples

TriX is intentionally falsifiable. The repo includes explicit counterexamples that draw the boundary between what is guaranteed and what is not.

## Routing Equivalences

- Dot vs XOR+POPCNT is not universally equivalent for ternary inputs with zeros unless masking is applied.
  - See `docs/DOT_POPCOUNT_EQUIVALENCE.md`
  - Tests: `tests/test_xor_superposition.py`

## Tie / Near-Tie Degeneracy

- Collapse and instability are not universal properties of argmax routing.
- Tie-degenerate geometries can collapse to a single tile and can exhibit high churn under small perturbations.
  - Native harness + falsification: `native/` + `native/tests/test_falsify.cpp`

## Lifecycle Telemetry Semantics

- For hierarchical routers, margin/tie telemetry can be cluster-local and may not reflect a global over-all-tiles decision.
  - See `docs/LIFECYCLE_V1.md`
  - Tests: `tests/test_lifecycle_falsify.py`

## Alpha Scales (BitNet-style)

- Ternary+alpha semantics are correct relative to the declared computation, but alpha is not a universal approximation guarantee for float weights.
- Mean(abs) alpha can be pathological (sparse outlier dilution) and quantization is discontinuous near threshold.
  - See `docs/ALPHA_SCALES.md`
- Tests: `tests/test_kernel_alpha_falsify.py`

## CLI

- `trix bench` runs the benchmark suite under `experiments/`.
  - In wheel-only installs without the repo checkout, `experiments/` is typically not available.
  - In that environment, `trix bench` is expected to fail with a clear message.
