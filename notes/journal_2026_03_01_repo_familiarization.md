# Journal (2026-03-01) - Repo Familiarization

Context: First pass getting oriented in the local `trix` repo, identifying what feels genuinely differentiating vs prior art, then validating the engineering surface by running the full test suite.

## Novel Bits (As Implemented Here)

1) Routing as an operable artifact (not just an internal)
- The repo treats routing as something you can inspect, measure, edit ("surgery"), regularize, and later lock down.
- This shows up as a deliberate API surface (claims, stats, history) rather than just a training trick.
- Pointers: `src/trix/nn/sparse_lookup_v2.py`, `docs/SPARSE_LOOKUP_V2_API.md`

2) "Routing as a contract" via compilation
- Profile class->tile behavior, compile stable classes to an explicit dispatch table, and add guards/fallbacks.
- This is closer to deployment engineering than typical MoE repos.
- Pointers: `src/trix/nn/compiled_dispatch.py`, `tests/test_compiled_dispatch.py`

3) Circuit compiler pipeline (spec->decompose->verify->compose->emit)
- Ambitious framing: if atoms are verified and wiring is correct, system correctness is inherited.
- FP4 mode is particularly interesting: "constructed" threshold-circuit atoms instead of training.
- Pointers: `src/trix/compiler/`, `src/trix/compiler/README.md`

4) XOR signature compression as a first-class system concern
- XOR superposition (centroid + sparse deltas) applied to signature storage/routing is a tight idea.
- Primitives are known (XOR deltas, Hamming distance), but the packaging toward deterministic routing is distinctive.
- Pointers: `CHANGELOG.md` (Mesa 13), `src/trix/nn/xor_routing.py`

## What Running Tests Taught Us

Baseline result:
- After environment/portability fixes, `pytest` completes with: 316 passed, 12 skipped.

The failures were not conceptual; they were portability assumptions:
- Native kernel loader assumed a Linux-style `libtrix.so` even on macOS.
- Number theory tests assumed optional dependency `gmpy2` was installed.

Fixes made to get clean signal from the suite:
- `src/trix/kernel/bindings.py`
  - Search for platform-appropriate shared library names (e.g. `.dylib` on macOS)
  - Provide Python fallbacks for `pack_weights`, `unpack_weights`, and `trix_forward` when the native lib is missing
- `tests/test_number_theory.py`
  - Skip GMP-backed tests when `gmpy2` is not installed (rather than erroring on import)

Most useful "this is real" signals from the tests:
- Compiled dispatch behavior is exercised as an invariant-driven feature, not a demo.
- SparseLookup v2 has a broad test surface, suggesting the v2 API is treated as stable-ish.

## Versioning Hygiene (Fixed)

Observed mismatch:
- Package metadata and runtime `__version__` were inconsistent, and did not match the top entry in `CHANGELOG.md`.

Aligned to a single source of truth for the root package:
- `pyproject.toml` and `src/trix/__init__.py` now agree.

Note: `TriXO/` and `TriXOR/` appear to be sibling variants with their own `pyproject.toml`; their runtime `__version__` values were updated to match their local metadata.

## Open Questions / Follow-ups

- Decide what "the" canonical version stream is (root vs `TriXO/` vs `TriXOR/`) and whether the siblings should be separate distribution names to avoid ambiguity.
- Consider adding an explicit optional extra for number theory (e.g. `trix[number-theory]`) to formalize `gmpy2` instead of implicit optional behavior.
- Consider registering pytest markers (e.g. `slow`) to reduce warning noise and make CI intent clear.

## Potential Applications

- Transformer FFN variants with conditional compute (tile/expert sparsity) for smaller memory footprint and potentially faster inference when activation is sparse: `src/trix/nn/`
- Debuggable specialization: inspect which inputs/classes route to which tiles, track claims, and manually edit/freeze behavior ("surgery") for known concepts: `src/trix/nn/sparse_lookup_v2.py`
- Deterministic, deployable routing via compilation: profile class->tile behavior, compile stable classes into a dispatch table with guardrails + fallback: `src/trix/nn/compiled_dispatch.py`
- Quantization + packing pipeline for edge/CPU inference (2-bit/ternary), with an optional native acceleration path: `src/trix/qat/`, `src/trix/kernel/`
- Boolean-ish / bounded-domain exact computation as "neural circuits": compile small circuits (adders/mux/etc.) into verified compositions; FP4 atoms are interesting here: `src/trix/compiler/`
- Signature compression + storage for large tile libraries (XOR delta/superposition, Hamming-space routing) as a systems trick for scaling routing metadata: `src/trix/nn/xor_routing.py`, `src/trix/hsquares_os/`
- Research sandbox for "routing as computation" and temporal/stateful routing ideas (plus associated experiments): `src/trix/nn/temporal_tiles.py`, `experiments/`, `docs/`

## Falsification Findings (Routing Primitive)

We added explicit falsification tests (native C++) to satisfy skeptics and to draw crisp boundaries around what is and is not guaranteed.

1) Non-collapse is not universal
- Counterexample: identical signatures across all tiles.
  - All dot scores tie; argmax tie-break collapses everything to tile 0.
- Counterexample: all-zero signatures.
  - All dot scores are 0; also collapses to tile 0.
- Takeaway: non-collapse requires distribution-health guards and/or regularizers; it is not implied by argmax(dot).

2) Stability under small perturbations is not universal
- Counterexample: tie-degenerate geometry (all-zero signatures).
  - Before perturbation: always routes to tile 0.
  - After tiny per-element signature resampling noise (e.g. flip_prob=0.001): churn can become large.
- Takeaway: stability claims must specify assumptions about margin/tie structure and signature diversity.

3) argmax(dot) != argmin(hamming) universally for ternary (with zeros)
- Counterexample exists where dot-product prefers one signature but bit-Hamming on 2-bit ternary codes prefers another.
- Takeaway: any dot<->Hamming equivalence must be stated with precise constraints (e.g. binary +/-1 only, or a modified distance that ignores x==0 positions).

Artifacts:
- `docs/ADDRESS_CONTRACT.md` documents these known counterexamples.
- Native falsification tests live in `native/tests/test_falsify.cpp` and are run via `ctest`.

Follow-up (tightened claim + guards):
- We wrote a constrained equivalence note and test harness for when dot-product routing can be implemented as packed XOR+POPCNT distance:
  - `docs/DOT_POPCOUNT_EQUIVALENCE.md`
  - `native/tests/test_dot_popcount_equivalence.cpp`
- We extended native telemetry with tie/margin fields and added a deterministic tie-break (`--tie-break hash`) plus an optional tie-guard (`--guard-ties`) for tie-degenerate geometries.
