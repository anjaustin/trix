# Native Tools

This directory contains C++ utilities that treat routing as a primitive and emit lightweight telemetry.

Targets:
- `trix_routebench`: routing microbench + stability (churn) benchmark

Build (macOS / Apple Silicon):

```bash
cmake -S native -B native/build
cmake --build native/build -j
ctest --test-dir native/build
```

Run:

```bash
./native/build/trix_routebench --benchmark routing --tiles 64 --dim 512 --inputs 4096 --seed 1
./native/build/trix_routebench --benchmark stability --tiles 64 --dim 512 --inputs 4096 --seed 1 --flip-prob 0.01

# Degeneracy detector: churn vs margin under near-tie regimes
./native/build/trix_routebench --benchmark margin_sweep --tiles 2 --dim 1024 --inputs 8192 --diff-dims 15 --bias 0.55 --flip-prob 0.001

Notes:
- Use an odd `--diff-dims` to avoid exact ties in the bias sweep (tie_rate ~ 0), so you measure churn vs near-tie margins rather than tie-breaking behavior.
```

Tie handling:

```bash
# Deterministic tie-break (default: first)
./native/build/trix_routebench --benchmark routing --tie-break hash

# Guard against tie-degenerate collapse (switches first->hash if ties occur)
./native/build/trix_routebench --benchmark routing --tie-break first --guard-ties
```

JSONL telemetry:

```bash
./native/build/trix_routebench --benchmark routing --tiles 64 --dim 512 --inputs 4096 --seed 1 --jsonl results.jsonl
```

Notes:
- Signatures and inputs are generated as ternary int8 vectors in {-1,0,+1}.
- Routing is argmax over dot products (a minimal "routing as primitive" reference kernel).
- The stability benchmark perturbs signatures with per-element resampling noise and reports route churn.
