# Docs Index

This repo contains a mix of:
- practical "how to run it" documentation,
- API references,
- and research journals/speculation.

This index points you to the highest-signal docs first.

## Start Here

- `README.md` (repo overview, install, tests, pointers)
- `docs/INSTALL.md` (setup, optional native builds)
- `docs/QUICKSTART.md` (SparseLookup v2 + CompiledDispatch walkthrough)

## Routing As A Primitive (Engineering)

- `PRD.md` (current direction)
- `docs/ADDRESS_CONTRACT.md` (address invariants + telemetry schema)
- `docs/DOT_POPCOUNT_EQUIVALENCE.md` (exact conditions for dot vs XOR+POPCNT equivalence)
- `docs/LIFECYCLE_V1.md` (lifecycle wrapper semantics and falsification notes)
- `native/README.md` (C++ routing benchmark + falsification + degeneracy detector)

## API References

- `docs/SPARSE_LOOKUP_V2_API.md`

## Native / Kernel Notes

- `docs/NEON_OPTIMIZATION_PLAN.md`
- `src/trix/kernel/` (CMake build + bindings)

## Experiments

Experiments are intentionally separate from the core test surface and may require extra dependencies.

- `experiments/benchmarks/benchmark_v2_rigorous.py`
- `experiments/cpu_6502/trix_6502_v2_organs.py`
- `experiments/fft_atoms/` (FFT atom work)

## Research Notes

Many `docs/*.md` files are session logs, synthesis notes, and exploratory writeups. Treat them as archival context unless they are explicitly referenced by the docs above.
