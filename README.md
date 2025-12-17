# TriX (Ternary Routing with Isomorphic eXecution)

A 2-bit conditional ternary FFN for transformers with **learned computational sparsity** via **emergent routing**.
> Core idea: **Don’t learn what you can read.**
## Why TriX?

TriX is a drop-in replacement for transformer FFN layers that aims to deliver:

- **16× memory compression** (2-bit packed weights; 4 weights/byte)
- **Sparse compute** (only the winning tile computes per input)
- **Zero routing parameters** (routing emerges from weight structure) 
- Reported **quality gain** on TinyShakespeare char-LM (see Results) 

## What's New in v0.8.0

**The Neural CUDA Release** - SASS assembly execution on the TriX architecture.

| Mesa | Capability | What It Enables |
|------|------------|-----------------|
| **Mesa 1** | Discovery | Tiles specialize without supervision (92% purity on 6502 ops) |
| **Mesa 2** | Partnership | Surgery API, claim tracking, regularizers |
| **Mesa 3** | Compilation | O(1) dispatch for known classes |
| **Mesa 4** | Temporal Binding | State routing replaces attention (100% bracket counting) |
| **Mesa 5** | FFT/WHT | Twiddle opcodes, 0.00 error vs NumPy |
| **Mesa 6** | Butterfly MatMul | Monarch structures, 0.00 error |
| **Mesa 7** | Isomorphic Transformer | SpectralMixer + ButterflyMLP |
| **Mesa 8** | Neural CUDA | SASS opcodes on TriX (100% exact) |

### Mesa 8: The Neural GPU

```
SASS Opcode → TriX Router → Tile → FP4 Atoms → Exact Result
     ↓            ↓          ↓         ↓           ↓
  IADD3     Signature    INTEGER   SUM+CARRY     100
            Matching      _ALU      atoms
```

- **Routing**: Ternary signature matching dispatches opcodes to tiles
- **Tiles**: RippleAdderTile composed of FullAdders
- **Atoms**: SUM (parity) + CARRY (majority) - FP4 threshold circuits
- **Result**: 42 + 58 = 100 (exact, through full TriX stack)

### The Unified Architecture

| Mesa | Domain | Cartridge | Status |
|------|--------|-----------|--------|
| 5 | Signal Processing | Twiddle Opcodes | ✅ 0.00 error |
| 6 | Linear Algebra | Block Opcodes | ✅ 0.00 error |
| 8 | General Purpose | SASS Opcodes | ✅ 100% exact |

**One engine. Every cartridge. Universal computation.**

```python
# Spatial routing (Mesa 1-3)
from trix.nn import SparseLookupFFNv2, CompiledDispatch

# Train with claim tracking
ffn = SparseLookupFFNv2(d_model=128, num_tiles=16, ternary_weight=0.01)
output, info, aux = ffn(x, labels=class_labels)

# Compile stable classes
compiler = CompiledDispatch(ffn)
compiler.compile_stable(threshold=0.5)

# Deploy with O(1) dispatch
output, info, aux = compiler.forward(x, class_hint=0, confidence=0.9)

# Temporal routing (Mesa 4)
from trix.nn import TemporalTileLayer

temporal = TemporalTileLayer(d_model=32, d_state=16, num_tiles=8)
state = temporal.init_state(batch_size=4)
output, final_state, infos = temporal.forward_sequence(x)
# Tiles learn state transitions - the counter emerges from routing

# Complete FFT Subsystem (Mesa 5)
# See experiments/fft_atoms/ for full implementation:
#   - pure_trix_fft_discrete.py: Real FFT (100%)
#   - pure_trix_fft_twiddle_v2.py: Complex FFT with twiddles (100%)
#   - pure_trix_fft_nscale_v2.py: N-scaling 8→64 (100%)
#   - pure_trix_fft_ifft.py: Forward/Inverse closure (100%)
```

See [QUICKSTART.md](docs/QUICKSTART.md) for the full tutorial.

## Status / Hardware support

- ✅ **Tested:** Jetson AGX Thor (current dev target) 
- ⚠️ **Untested:** other CUDA GPUs, CPU-only, macOS (PRs welcome)

> NOTE: README previously said “Jeston” — should be “Jetson”. 

## Install

> TriX depends only on Python + PyTorch + NumPy. 

1) Install PyTorch for your platform (Jetson/CUDA/CPU).
2) Then:


```bash
pip install -e .
```

## Quick start

### SparseLookupFFN (recommended)

```py
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

> *Wisdom is knowing when not to compute.* — SparseLookup uses routing to select directions and splines to modulate magnitude. No matrix multiplies in the hot path.

### HierarchicalTriXFFN (original)

```py
import torch
from trix import HierarchicalTriXFFN

x = torch.randn(2, 128, 512)  # (batch, seq, d_model)

ffn = HierarchicalTriXFFN(
    d_model=512,
    num_tiles=16,
    tiles_per_cluster=4,
)

output, routing_info, aux_losses = ffn(x)

loss = some_task_loss(output) + aux_losses["total_aux"]
```

See: `examples/nvidia_quickstart.py` for plug-and-play usage. ([GitHub][1])

## How it works (high level)

### 1) Ternary weights = votes

Each weight is in `{ -1, 0, +1 }`:

* `+1`: “want this feature”
* `-1`: “want the opposite”
* `0`: “don’t care” ([GitHub][1])

### 2) Tile signatures = addresses

Each tile gets a signature derived from its weights:

```py
signature = weights.sum(dim=0).sign()
```

### 3) Routing = content lookup

Inputs route to the tile whose signature matches best:

```py
scores = input @ signatures.T
winner = scores.argmax()
```

No learned router, no router params — just structure. ([GitHub][1])

## Key components

| Component               | Use case                                                  |
| ----------------------- | --------------------------------------------------------- |
| `TemporalTileLayer`     | **v0.5.4:** State routing for temporal binding |
| `SparseLookupFFNv2`     | Surgery, claim tracking, regularizers |
| `CompiledDispatch`      | Path compilation for O(1) dispatch |
| `SparseLookupFFN`       | Routing IS computation, 2.3× smaller |
| `HierarchicalTriXFFN`   | FFN with hierarchical routing                             |
| `HierarchicalTriXBlock` | Full transformer block                                    |
| `SparseTriXFFN`         | Simple 4-tile FFN                                         |
| `TriXLinear`            | Low-level ternary linear                                  |

## Results

Validated on **TinyShakespeare character-level language modeling**.

| Model                | Params  | Val PPL |      vs baseline |
| -------------------- | ------: | ------: | ---------------: |
| Sparse-4tiles        |     —   |   19.26 |                — |
| Hierarchical-16tiles | 826,304 |   17.16 |           −10.9% |
| **SparseLookup-64**  | **366,412** | **16.56** | **−14.0%** |

> **SparseLookupFFN**: 2.3× fewer parameters, best perplexity. *Routing IS the computation.*

### Reproduce

```bash
# Run the benchmark (compares all FFN types)
python scripts/benchmark_ffn.py
```

* **Dataset:** TinyShakespeare (auto-downloaded)
* **Config:** d_model=128, n_layers=4, num_tiles=64, tiles_per_cluster=8
* **Seed:** 42
* **Epochs:** 10 (20k training samples)

Expected output: SparseLookupFFN achieves lowest PPL with fewest parameters.

## Project layout

```text
src/trix/
  nn/              # modules (SparseLookup / hierarchical / sparse / classic)
  kernel/          # 2-bit kernel with ARM NEON
  qat/             # quantization-aware training
tests/             # 268 tests
examples/          # usage examples
experiments/
  fft_atoms/       # Mesa 5: FFT atom tests and hybrid architecture
scripts/           # benchmark and validation scripts
notes/             # design exploration and process docs
docs/              # architecture docs and research notes
```

(See `docs/BUILD_LOG.md`, `docs/ABSTRACT.md`, `docs/BIG_LEAP_SPEC.md`.) ([GitHub][1])

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and migration guides.

## License

MIT.

## `scripts/smoke_trix.py` (simple “does it run?” test)

```py
#!/usr/bin/env python3
"""
Smoke test for TriX modules.

Goal:
- import works
- forward pass works
- output shape matches input shape
- aux_losses contains total_aux
- routing_info is present (even if structure varies)
"""

import sys
import torch

def main() -> int:
    try:
        from trix import HierarchicalTriXFFN
    except Exception as e:
        print("❌ Import failed: from trix import HierarchicalTriXFFN")
        print(e)
        return 1

    torch.manual_seed(0)

    # Keep this small so it runs everywhere (CPU included)
    B, T, D = 2, 16, 64
    x = torch.randn(B, T, D)

    ffn = HierarchicalTriXFFN(
        d_model=D,
        num_tiles=8,
        tiles_per_cluster=4,
    )

    try:
        out, routing_info, aux_losses = ffn(x)
    except Exception as e:
        print("❌ Forward pass failed")
        print(e)
        return 2

    ok = True

    if not isinstance(out, torch.Tensor):
        print("❌ output is not a torch.Tensor")
        ok = False
    else:
        print(f"✅ output: {tuple(out.shape)}")
        if tuple(out.shape) != tuple(x.shape):
            print(f"❌ expected output shape {tuple(x.shape)}")
            ok = False

    # routing_info: allow any type, but it must exist
    print(f"✅ routing_info type: {type(routing_info).__name__}")

    if not isinstance(aux_losses, dict):
        print("❌ aux_losses is not a dict")
        ok = False
    else:
        keys = sorted(aux_losses.keys())
        print(f"✅ aux_losses keys: {keys}")
        if "total_aux" not in aux_losses:
            print("❌ aux_losses missing 'total_aux'")
            ok = False
        else:
            ta = aux_losses["total_aux"]
            if isinstance(ta, torch.Tensor):
                print(f"✅ total_aux: {ta.item() if ta.numel()==1 else ta.shape}")
            else:
                print(f"✅ total_aux: {ta}")

    print("✅ SMOKE PASS" if ok else "❌ SMOKE FAIL")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
````

Run it:

```bash
python scripts/smoke_trix.py
```

---

[1]: https://github.com/anjaustin/trix "GitHub - anjaustin/trix: A 2-Bit Conditional Ternary Neural Architecture with Learned Computational Sparsity"
