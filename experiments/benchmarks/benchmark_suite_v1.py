#!/usr/bin/env python3
"""Benchmark suite v1.

This is a small, CPU-first suite intended to make TriX claims falsifiable.

It produces:
- a single JSON summary artifact
- an optional JSONL telemetry stream (routing/stability events)

Run:
  python experiments/benchmarks/benchmark_suite_v1.py --outdir results/benchmarks_v1
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_imports() -> None:
    try:
        import trix  # noqa: F401

        return
    except Exception:
        pass

    root = _repo_root()
    sys.path.insert(0, str(root / "src"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_jsonable(x: Any) -> Any:
        if x is None or isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (list, tuple)):
            return [to_jsonable(v) for v in x]
        if isinstance(x, dict):
            return {str(k): to_jsonable(v) for k, v in x.items()}
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return float(x.item())
            return x.detach().cpu().tolist()
        # numpy / torch scalar-like
        if hasattr(x, "item") and callable(getattr(x, "item")):
            try:
                v = x.item()
                return to_jsonable(v)
            except Exception:
                pass
        return str(x)

    path.write_text(
        json.dumps(to_jsonable(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _env_info() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available()
        if hasattr(torch.backends, "mps")
        else False,
        "cuda_available": torch.cuda.is_available(),
    }


def _bench_fp4_atoms(*, include_slow: bool) -> Dict[str, Any]:
    _ensure_imports()
    from trix.compiler.atoms_fp4 import FP4AtomLibrary

    out: Dict[str, Any] = {
        "name": "fp4_atoms",
        "ok": True,
        "slow_included": bool(include_slow),
        "timing_s": {},
        "details": {},
    }

    t0 = time.perf_counter()
    lib = FP4AtomLibrary()

    expected = ["AND", "OR", "XOR", "NOT", "NAND", "NOR", "XNOR", "SUM", "CARRY", "MUX"]
    missing = []
    not_verified = []

    for name in expected:
        atom = lib.get_atom(name)
        if atom is None:
            missing.append(name)
            continue
        if getattr(getattr(atom, "status", None), "name", None) != "VERIFIED":
            not_verified.append(name)

    out["details"]["missing"] = missing
    out["details"]["not_verified"] = not_verified
    if missing or not_verified:
        out["ok"] = False

    # Exhaustive truth tables for a representative subset
    try:
        and_atom = lib.get_atom("AND")
        or_atom = lib.get_atom("OR")
        xor_atom = lib.get_atom("XOR")
        sum_atom = lib.get_atom("SUM")
        carry_atom = lib.get_atom("CARRY")

        # AND/OR/XOR
        table2 = {
            "AND": {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
            "OR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1},
            "XOR": {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
        }

        for (a, b), exp in table2["AND"].items():
            y = and_atom(torch.tensor([[float(a), float(b)]]))
            got = int(y[0, 0].item() > 0.5)
            if got != exp:
                raise AssertionError(f"AND({a},{b})={got} expected {exp}")

        for (a, b), exp in table2["OR"].items():
            y = or_atom(torch.tensor([[float(a), float(b)]]))
            got = int(y[0, 0].item() > 0.5)
            if got != exp:
                raise AssertionError(f"OR({a},{b})={got} expected {exp}")

        for (a, b), exp in table2["XOR"].items():
            y = xor_atom(torch.tensor([[float(a), float(b)]]))
            got = int(y[0, 0].item() > 0.5)
            if got != exp:
                raise AssertionError(f"XOR({a},{b})={got} expected {exp}")

        # SUM/CARRY for all 3-bit combos
        errors = 0
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    x = torch.tensor([[float(a), float(b), float(c)]])
                    s = int(sum_atom(x)[0, 0].item() > 0.5)
                    co = int(carry_atom(x)[0, 0].item() > 0.5)
                    exp_s = (a + b + c) % 2
                    exp_c = 1 if (a + b + c) >= 2 else 0
                    if s != exp_s or co != exp_c:
                        errors += 1
        out["details"]["sum_carry_errors"] = errors
        if errors:
            out["ok"] = False
    except Exception as e:
        out["ok"] = False
        out["details"]["truth_table_error"] = repr(e)

    out["timing_s"]["atoms_total"] = time.perf_counter() - t0

    # Optional: 8-bit adder exhaustive (slow)
    if include_slow:
        t1 = time.perf_counter()
        try:
            sum_atom = lib.get_atom("SUM")
            carry_atom = lib.get_atom("CARRY")

            def add_8bit(a: int, b: int) -> int:
                carry = 0
                outv = 0
                for i in range(8):
                    ai = (a >> i) & 1
                    bi = (b >> i) & 1
                    x = torch.tensor([[float(ai), float(bi), float(carry)]])
                    s = int(sum_atom(x)[0, 0].item() > 0.5)
                    carry = int(carry_atom(x)[0, 0].item() > 0.5)
                    outv |= s << i
                return outv

            errors = 0
            for a in range(256):
                for b in range(256):
                    exp = (a + b) & 0xFF
                    got = add_8bit(a, b)
                    if got != exp:
                        errors += 1
                        if errors > 10:
                            break
                if errors > 10:
                    break

            out["details"]["adder8_errors"] = errors
            if errors:
                out["ok"] = False
        except Exception as e:
            out["ok"] = False
            out["details"]["adder8_error"] = repr(e)
        out["timing_s"]["adder8_exhaustive"] = time.perf_counter() - t1
    else:
        out["details"]["adder8_exhaustive"] = "skipped"

    return out


def _bench_routing_compiled_dispatch(
    *,
    seed: int,
    device: str,
    outdir: Path,
) -> Dict[str, Any]:
    _ensure_imports()
    from trix.nn import SparseLookupFFNv2, RoutingLifecycleV1
    from trix.nn.compiled_dispatch import CompiledDispatch

    torch.manual_seed(seed)
    dev = torch.device(device)

    d_model = 64
    num_tiles = 8
    tiles_per_cluster = 4
    num_classes = 5

    # A stable class->tile mapping (mirrors test_compiled_dispatch fixture intent).
    class_to_tile = {0: 2, 1: 5, 2: 0, 3: 1, 4: 7}

    ffn = SparseLookupFFNv2(
        d_model=d_model,
        num_tiles=num_tiles,
        tiles_per_cluster=tiles_per_cluster,
        dropout=0.0,
        use_score_calibration=False,
    ).to(dev)

    # Create per-class prototypes and surgically install signatures.
    prototypes: Dict[int, torch.Tensor] = {}
    for c, tile in class_to_tile.items():
        p = torch.randn(d_model, device=dev)
        p = (p - p.mean()) / (p.std() + 1e-6)
        prototypes[c] = p

        sig = torch.zeros(d_model, device=dev)
        sig[p > 0.2] = 1.0
        sig[p < -0.2] = -1.0
        ffn.insert_signature(tile, sig.detach().cpu(), freeze=True, tag=f"class_{c}")

    # Build a dataset: sequences are class-homogeneous for clean class_hint.
    B = 64
    T = 16
    steps = 8
    noise = 0.25

    def make_batch(step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = torch.empty((B, T, d_model), device=dev)
        labels = torch.empty((B, T), dtype=torch.long, device=dev)
        classes = torch.tensor([(i + step) % num_classes for i in range(B)], device=dev)
        for i in range(B):
            c = int(classes[i].item())
            base = prototypes[c]
            xs[i] = base + noise * torch.randn((T, d_model), device=dev)
            labels[i].fill_(c)
        return xs, labels

    # Populate claim_matrix with a few training-mode passes.
    ffn.train()
    ffn.reset_stats()
    for s in range(steps):
        x, y = make_batch(s)
        _out, _info, _aux = ffn(x, labels=y)

    compiler = CompiledDispatch(ffn)
    compiled = compiler.compile_stable(
        threshold=0.3, min_confidence=0.0, num_classes=num_classes
    )

    # Evaluate routing purity on a fresh batch.
    ffn.eval()
    x_eval, y_eval = make_batch(999)
    with torch.no_grad():
        B2, T2, D2 = x_eval.shape
        x_flat = ffn.norm(x_eval).view(-1, D2)
        tile_idx, _scores = ffn.route(x_flat, ffn.signatures, return_scores=False)
        tile_idx = tile_idx.view(B2, T2)

    purity: Dict[int, float] = {}
    for c, tile in class_to_tile.items():
        mask = y_eval == c
        if mask.any():
            purity[c] = (tile_idx[mask] == tile).float().mean().item()
        else:
            purity[c] = None

    # Validate compiled execution semantics: compiled output equals forced-tile output.
    compiler.reset_stats()
    max_abs_err = 0.0
    with torch.no_grad():
        for c, tile in class_to_tile.items():
            x_c = x_eval[y_eval[:, 0] == c]
            if x_c.numel() == 0:
                continue
            out_c, info_c, _aux_c = compiler.forward(x_c, class_hint=c, confidence=1.0)
            out_f, info_f, _aux_f = ffn.forward_forced_tile(x_c, tile_idx=tile)
            err = (out_c - out_f).abs().max().item()
            max_abs_err = max(max_abs_err, float(err))
            # Ensure compiled path was taken when compiled.
            if c in compiled:
                assert info_c.get("compiled") is True
                assert int(info_c.get("compiled_class")) == c
                assert torch.equal(info_c["tile_idx"], info_f["tile_idx"])

    stats = compiler.get_stats()

    # Emit telemetry via lifecycle.
    telemetry_path = outdir / "routing_telemetry.jsonl"
    lc = RoutingLifecycleV1(ffn, compiled=compiler)
    _out, _info, _aux, rec = lc.observe(
        x_eval[:8],
        class_hint=int(y_eval[0, 0].item()),
        confidence=1.0,
        jsonl_path=str(telemetry_path),
        run_id="bench_v1_routing",
    )
    stab = lc.stability_probe(
        x_eval[:8],
        flip_prob=0.001,
        seed=seed,
        jsonl_path=str(telemetry_path),
        run_id="bench_v1_stability",
    )

    return {
        "name": "routing_compiled_dispatch",
        "ok": True,
        "config": {
            "seed": seed,
            "device": str(dev),
            "d_model": d_model,
            "num_tiles": num_tiles,
            "tiles_per_cluster": tiles_per_cluster,
            "num_classes": num_classes,
            "class_to_tile": {str(k): int(v) for k, v in class_to_tile.items()},
            "steps": steps,
            "batch": {"B": B, "T": T, "noise": noise},
        },
        "purity": {str(k): v for k, v in purity.items()},
        "compiled": {
            "compiled_classes": sorted([int(k) for k in compiled.keys()]),
            "dispatch_table": compiler.export_dispatch_table(),
            "stats": stats,
        },
        "semantic_check": {"max_abs_err_compiled_vs_forced": max_abs_err},
        "telemetry": {
            "routing_record": rec,
            "stability_record": stab,
            "jsonl": str(telemetry_path),
        },
    }


def _bench_drift_under_regularizer_training(
    *,
    seed: int,
    device: str,
    outdir: Path,
) -> Dict[str, Any]:
    """Measure routing drift and contract drift under optimization.

    SparseLookupFFNv2 signature parameters are updated directly by the
    regularizers (ternary/sparsity/diversity). This benchmark intentionally
    optimizes the regularizer objective and measures how routing changes over
    time on a fixed evaluation batch.
    """

    _ensure_imports()
    from trix.nn import SparseLookupFFNv2, RoutingLifecycleV1
    from trix.nn.compiled_dispatch import CompiledDispatch

    torch.manual_seed(seed)
    dev = torch.device(device)

    d_model = 64
    num_tiles = 8
    tiles_per_cluster = 4
    num_classes = 5

    ffn = SparseLookupFFNv2(
        d_model=d_model,
        num_tiles=num_tiles,
        tiles_per_cluster=tiles_per_cluster,
        dropout=0.0,
        use_score_calibration=False,
        ternary_weight=0.05,
        sparsity_weight=0.05,
        diversity_weight=0.05,
    ).to(dev)

    B, T = 64, 16
    prototypes = torch.randn(num_classes, d_model, device=dev)
    prototypes = (prototypes - prototypes.mean(dim=-1, keepdim=True)) / (
        prototypes.std(dim=-1, keepdim=True) + 1e-6
    )

    def make_batch(step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = torch.empty((B, T, d_model), device=dev)
        labels = torch.empty((B, T), dtype=torch.long, device=dev)
        classes = torch.tensor([(i + step) % num_classes for i in range(B)], device=dev)
        for i in range(B):
            c = int(classes[i].item())
            xs[i] = prototypes[c] + 0.25 * torch.randn((T, d_model), device=dev)
            labels[i].fill_(c)
        return xs, labels

    x_eval, y_eval = make_batch(0)

    ffn.eval()
    with torch.no_grad():
        _o, info0, _aux0 = ffn(x_eval)
        base_tiles = info0["tile_idx"].detach().clone()

    # Populate claims.
    ffn.train()
    ffn.reset_stats()
    for s in range(4):
        x, y = make_batch(s)
        _o, _info, _aux = ffn(x, labels=y)

    compiler = CompiledDispatch(ffn)
    compiled = compiler.compile_stable(
        threshold=0.3, min_confidence=0.0, num_classes=num_classes
    )
    dispatch0 = compiler.export_dispatch_table()

    opt = torch.optim.Adam(ffn.parameters(), lr=1e-2)
    steps = 15

    churn = []
    drifted = []
    hit_rates = []
    total_aux = []

    lc = RoutingLifecycleV1(ffn, compiled=compiler)
    telemetry_path = outdir / "drift_telemetry.jsonl"

    prev_tiles = base_tiles
    for step in range(steps):
        x, y = make_batch(step)
        out, info, aux = ffn(x, labels=y)

        loss = aux["total_aux"]
        total_aux.append(float(loss.detach().cpu().item()))

        opt.zero_grad()
        loss.backward()
        opt.step()

        ffn.eval()
        with torch.no_grad():
            _o, info_e, _aux_e = ffn(x_eval)
            tiles = info_e["tile_idx"].detach()
            churn.append(float((tiles != prev_tiles).float().mean().item()))
            prev_tiles = tiles

            drifted_classes = compiler.check_drift(threshold=0.2)
            drifted.append([int(k) for k in drifted_classes])

            hit_rates.append(float(compiler.get_stats().get("hit_rate", 0.0)))

            lc.observe(
                x_eval[:4],
                class_hint=int(y_eval[0, 0].item()),
                confidence=1.0,
                jsonl_path=str(telemetry_path),
                run_id=f"drift_step_{step}",
            )

        ffn.train()

    return {
        "name": "drift_under_regularizer_training",
        "ok": True,
        "config": {
            "seed": seed,
            "device": str(dev),
            "steps": steps,
            "d_model": d_model,
            "num_tiles": num_tiles,
            "tiles_per_cluster": tiles_per_cluster,
            "num_classes": num_classes,
        },
        "compiled": {
            "compiled_classes": sorted([int(k) for k in compiled.keys()]),
            "dispatch_table_initial": dispatch0,
        },
        "metrics": {
            "total_aux": total_aux,
            "churn": churn,
            "drifted_classes": drifted,
            "compiled_hit_rate": hit_rates,
        },
        "telemetry": {"jsonl": str(telemetry_path)},
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results/benchmarks_v1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--include-slow", action="store_true")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_id = _now_run_id()

    suite: Dict[str, Any] = {
        "suite": "trix_benchmark_v1",
        "run_id": run_id,
        "env": _env_info(),
        "args": vars(args),
        "benchmarks": [],
    }

    t0 = time.perf_counter()
    suite["benchmarks"].append(_bench_fp4_atoms(include_slow=bool(args.include_slow)))
    suite["benchmarks"].append(
        _bench_routing_compiled_dispatch(
            seed=int(args.seed), device=str(args.device), outdir=outdir
        )
    )
    suite["benchmarks"].append(
        _bench_drift_under_regularizer_training(
            seed=int(args.seed), device=str(args.device), outdir=outdir
        )
    )
    suite["timing_s"] = {"total": time.perf_counter() - t0}

    ok = all(bool(b.get("ok")) for b in suite["benchmarks"])
    suite["ok"] = ok

    _write_json(outdir / "suite_v1.json", suite)
    print(f"wrote {outdir / 'suite_v1.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
