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
```

JSONL telemetry:

```bash
./native/build/trix_routebench --benchmark routing --tiles 64 --dim 512 --inputs 4096 --seed 1 --jsonl results.jsonl
```

Notes:
- Signatures and inputs are generated as ternary int8 vectors in {-1,0,+1}.
- Routing is argmax over dot products (a minimal "routing as primitive" reference kernel).
- The stability benchmark perturbs signatures with per-element resampling noise and reports route churn.
