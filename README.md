# TriX (Archived)

> **Active development has moved to [trix-z](../trix-z/).** This repo is preserved as the historical record with full git history, archived snapshots (TriXO/TriXOR), exploration journals, and mesa session notes.

See `ARCHIVED.md` for what changes belong here vs `../trix-z/`.

---

TriX is a research-oriented library for **routing-driven** transformer FFNs and related "compute-as-routing" experiments.

The practical throughline is treating routing as a **first-class artifact** you can:
- observe and measure,
- edit deliberately ("surgery"),
- compile into deterministic contracts for known cases,
- and accelerate natively without changing semantics.

Core idea: **Don't learn what you can read.**

Related concept note: `docs/SOFT_CHIP.md` (cache-resident subroutines / "soft chip" idea for training efficiency).

## What TriX Is

- A set of PyTorch modules under `src/trix/nn/` for sparse/conditional FFNs.
- A small native kernel under `src/trix/kernel/` for packing and fast CPU inference (optional).
- A compiler-style lane under `src/trix/compiler/` for composing verified "atoms" into circuits.

## What TriX Is Not

- A polished, production-ready framework.
- A single claim or benchmark; the repo includes experiments of varying maturity.

## Current Release

See `CHANGELOG.md`.

Recent work highlighted in the changelog:
- Mesa 13: XOR superposition signature compression (lossless routing decisions with large compression on similar signatures).

## Install

1) Install PyTorch for your platform.
2) Install TriX in editable mode:

```bash
pip install -e ".[dev]"
```

## Run Tests

```bash
python -m pytest
```

Notes:
- Native acceleration is optional. Tests pass without the native library.
- Some experiment tests (e.g. GMP-backed number theory) are skipped if optional deps are missing.

## P0-1 to P0-4: Repo Hygiene + Repro Journey

This repo is archived, but we still keep a CPU-first, falsifiable path that a new reader can run quickly.

- `P0-1` Archive historical snapshots: moved `TriXO/` + `TriXOR/` to `archive/` to prevent accidental `pip install -e .` namespace shadowing (they both declare `name = "trix"`).
- `P0-2` Fix doc link rot: repaired references and copied Mesa 11/12/13 docs into `docs/archive/` so older changelog entries remain navigable.
- `P0-3` Formalize tier boundaries: experiment-dependent tests now skip cleanly when `experiments/` (or optional deps) are absent.
- `P0-4` Add golden-output repro scripts: `scripts/repro/` turns key changelog claims into single-command checks that diff against saved `.expected.json` outputs.

Quick repro commands (CPU):

```bash
trix doctor
python -m pytest -q
trix bench --outdir results/benchmarks_v1 --device cpu

python scripts/repro/repro_xor_compression.py
python scripts/repro/repro_compiled_dispatch.py
python scripts/repro/repro_dft_compilation.py
```

Notes:
- `repro_dft_compilation.py` requires `experiments/fft_atoms/`; it prints `SKIP` in wheel installs.

## CLI

If installed in editable mode, you can use the `trix` CLI:

```bash
trix doctor
trix bench --outdir results/benchmarks_v1 --device cpu
```

See `docs/CLI.md`.

## Benchmarks

Canonical benchmark entrypoints live in `docs/BENCHMARKS.md`.

Run suite v1 (CPU-first, emits JSON + JSONL artifacts):

```bash
python experiments/benchmarks/benchmark_suite_v1.py --outdir results/benchmarks_v1 --device cpu
```

## Quick Start

### DropInFFN (recommended starting point)

If you're integrating TriX into an existing model, start here.

```python
import torch
from trix.nn import DropInFFN, DropInConfig

x = torch.randn(2, 128, 512)

ffn = DropInFFN(
    DropInConfig(d_model=512, num_tiles=64, tiles_per_cluster=8),
    mode="dynamic",  # or: contracted|packed
)

y = ffn(x)  # drop-in: returns a tensor

# Optional training signal
y, routing_info, aux_losses = ffn(x, return_aux=True)
loss = some_task_loss(y) + aux_losses["total_aux"]
```

### SparseLookupFFN (legacy quick start)

```python
import torch
from trix import SparseLookupFFN

x = torch.randn(2, 128, 512)  # (batch, seq, d_model)

ffn = SparseLookupFFN(
    d_model=512,
    num_tiles=64,
    tiles_per_cluster=8,
)

output, routing_info, aux_losses = ffn(x)
loss = some_task_loss(output) + aux_losses["total_aux"]
```

### Routing + Compilation (stable class dispatch)

```python
from trix.nn import SparseLookupFFNv2, CompiledDispatch

ffn = SparseLookupFFNv2(d_model=128, num_tiles=16, ternary_weight=0.01)
output, info, aux = ffn(x, labels=class_labels)

compiler = CompiledDispatch(ffn)
compiler.compile_stable(threshold=0.5)

output, info, aux = compiler.forward(x, class_hint=0, confidence=0.9)
```

See `docs/QUICKSTART.md` and `examples/basic_usage.py`.

## Pick Your Lane

- `dynamic`: research mode; routing is computed each forward
- `contracted`: compile stable classes into a dispatch contract and guard it
- `packed`: ternary address mode using XOR+POPCNT distances (inputs ternarized via `sign(x)`)

Developer-first entrypoints:
- `trix doctor` (self-check)
- `trix bench` (writes `suite_v1.json` + telemetry)
- `trix export-bundle` / `trix load-bundle` (portable address plane)

See `docs/CLI.md` and `docs/BUNDLE_SCHEMA.md`.

## Minimal Routing Primitive (Conceptual)

Many routing variants in this repo reduce to a simple kernel:

```python
signature = tile_weights.sum(dim=0).sign()  # What the tile wants
score = input @ signature                   # How well input matches
route = (score == scores.max())             # Send to best match
```

The work is in making this primitive stable and operable at scale (normalization, tie-breaking/top-k, load balancing, differentiable surrogates for training, and compilation/guardrails for deployment).

## Optional Native Kernel (CPU)

TriX includes a small C++ library used by `trix.kernel` for packing and a fast forward path.

Build:

```bash
cmake -S src/trix/kernel -B src/trix/kernel/build
cmake --build src/trix/kernel/build -j
```

Correctness note:
- When the native library is present, `tests/test_kernel_reference_harness.py` enforces native-vs-reference equivalence.

## Native Routing Tools (C++)

This repo also contains a standalone C++ routing benchmark harness under `native/`.

Build and run:

```bash
cmake -S native -B native/build
cmake --build native/build -j
./native/build/trix_routebench --benchmark routing
./native/build/trix_routebench --benchmark stability --flip-prob 0.01
```

See `native/README.md`.

## Results (Repro Script)

There is a benchmark script that compares FFN variants on TinyShakespeare:

```bash
python scripts/benchmark_ffn.py
```

Treat these numbers as a reproducible starting point, not as a universal claim.

## Project Layout

```text
src/trix/
  nn/              # routing-driven FFNs and related modules
  kernel/          # optional native kernel bindings + packing
  compiler/        # circuit compilation lane
  qat/             # quantization-aware training
tests/             # test suite
examples/          # usage examples
scripts/           # benchmarks and validations
experiments/       # research experiments (may require optional deps)
notes/             # process / journals / exploration
docs/              # documentation
native/            # standalone C++ routing tools
```

## Changelog

See `CHANGELOG.md` for version history.

## License

MIT.

[1]: https://github.com/anjaustin/trix "GitHub - anjaustin/trix"
