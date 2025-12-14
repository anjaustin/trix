````md
# TriX

A 2-bit conditional ternary FFN for transformers with **learned computational sparsity** via **emergent routing**. :contentReference[oaicite:1]{index=1}

> Core idea: **Don’t learn what you can read.** :contentReference[oaicite:2]{index=2}

## Why TriX?

TriX is a drop-in replacement for transformer FFN layers that aims to deliver:

- **16× memory compression** (2-bit packed weights; 4 weights/byte) :contentReference[oaicite:3]{index=3}
- **Sparse compute** (only the winning tile computes per input) :contentReference[oaicite:4]{index=4}
- **Zero routing parameters** (routing emerges from weight structure) :contentReference[oaicite:5]{index=5}
- Reported **quality gain** on TinyShakespeare char-LM (see Results) :contentReference[oaicite:6]{index=6}

## Status / Hardware support

- ✅ **Tested:** Jetson AGX Thor (current dev target) :contentReference[oaicite:7]{index=7}
- ⚠️ **Untested:** other CUDA GPUs, CPU-only, macOS (PRs welcome)

> NOTE: README previously said “Jeston” — should be “Jetson”. :contentReference[oaicite:8]{index=8}

## Install

> TriX depends only on Python + PyTorch + NumPy. :contentReference[oaicite:9]{index=9}

1) Install PyTorch for your platform (Jetson/CUDA/CPU).
2) Then:

```bash
pip install -e .
```

````

## Quick start

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
| `HierarchicalTriXFFN`   | FFN with hierarchical routing (recommended) ([GitHub][1]) |
| `HierarchicalTriXBlock` | Full transformer block ([GitHub][1])                      |
| `SparseTriXFFN`         | Simple 4-tile FFN ([GitHub][1])                           |
| `TriXLinear`            | Low-level ternary linear ([GitHub][1])                    |

## Results

Validated on **TinyShakespeare character-level language modeling**. ([GitHub][1])

| Model                | Val PPL |          vs baseline |
| -------------------- | ------: | -------------------: |
| Sparse-4tiles        |   19.26 |      — ([GitHub][1]) |
| Hierarchical-16tiles |   16.67 | −13.4% ([GitHub][1]) |

### Reproduce (fill this in)

> Add the exact commands/configs + expected output here.

* Dataset/source:
* Config:
* Seed:
* Train steps:
* Command:

```bash
python -m ...
```

## Project layout

```text
src/trix/
  nn/          # modules (hierarchical / sparse / classic)
  kernel/      # 2-bit kernel
  qat/         # quantization-aware training
tests/
examples/
docs/
```

(See `docs/BUILD_LOG.md`, `docs/ABSTRACT.md`, `docs/BIG_LEAP_SPEC.md`.) ([GitHub][1])

## License

MIT. ([GitHub][1])

````

---

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

If you want one extra “chef’s kiss” improvement: I can also sketch a tiny **GitHub Actions workflow** that runs this smoke test on CPU (so every PR proves the repo still imports + forwards), even if the Jetson kernel path is the primary target.

[1]: https://github.com/anjaustin/trix "GitHub - anjaustin/trix: A 2-Bit Conditional Ternary Neural Architecture with Learned Computational Sparsity"
