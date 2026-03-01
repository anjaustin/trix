# TriX (Ternary Routing with Isomorphic eXecution)

TriX is a research-oriented library for **routing-driven** transformer FFNs and related "compute-as-routing" experiments.

The practical throughline is treating routing as a **first-class artifact** you can:
- observe and measure,
- edit deliberately ("surgery"),
- compile into deterministic contracts for known cases,
- and accelerate natively without changing semantics.

Core idea: **Don't learn what you can read.**

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

## Benchmarks

Canonical benchmark entrypoints live in `docs/BENCHMARKS.md`.

Run suite v1 (CPU-first, emits JSON + JSONL artifacts):

```bash
python experiments/benchmarks/benchmark_suite_v1.py --outdir results/benchmarks_v1 --device cpu
```

## Quick Start

### SparseLookupFFN (recommended starting point)

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
