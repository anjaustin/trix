from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _repo_root_from_here() -> Optional[Path]:
    # Works in editable installs (repo layout). In wheels, experiments/docs may not exist.
    p = Path(__file__).resolve()
    for _ in range(6):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
        p = p.parent
    return None


def _print_kv(k: str, v: Any) -> None:
    print(f"{k}: {v}")


def _doctor() -> int:
    import trix
    from trix.kernel import (
        is_neon_available,
        pack_weights,
        unpack_weights,
        trix_forward,
    )
    from trix.kernel import bindings as b

    _print_kv("trix", trix.__version__)
    _print_kv("python", sys.version.split()[0])
    _print_kv("torch", torch.__version__)
    _print_kv("cuda_available", torch.cuda.is_available())
    if hasattr(torch.backends, "mps"):
        _print_kv("mps_available", torch.backends.mps.is_available())
    _print_kv("native_kernel_available", is_neon_available())

    # Optional deps
    for dep in ("scipy", "gmpy2"):
        try:
            __import__(dep)
            _print_kv(f"dep.{dep}", "present")
        except Exception:
            _print_kv(f"dep.{dep}", "missing")

    # Kernel correctness quick check
    torch.manual_seed(0)
    rows, cols = 17, 259
    w = torch.sign(torch.randn(rows, cols))
    w[0, 0] = 0.0
    packed = pack_weights(w)
    packed_py = b._pack_weights_python(w)
    if not torch.equal(packed.cpu(), packed_py.cpu()):
        print("FAIL: pack_weights != python reference")
        print("Next: run `python -m pytest -q tests/test_kernel_reference_harness.py`")
        return 2
    w2 = unpack_weights(packed, rows, cols)
    if not torch.equal(w.cpu(), w2.cpu()):
        print("FAIL: unpack_weights(pack_weights(w)) != w")
        print("Next: run `python -m pytest -q tests/test_kernel_reference_harness.py`")
        return 2

    batch, in_f, out_f, num_tiles = 2, 33, 64, 4
    x = torch.randn(batch, in_f)
    w3 = torch.sign(torch.randn(out_f, in_f))
    scales = torch.ones(out_f)
    gate = torch.randint(0, 2, (batch, num_tiles), dtype=torch.int64).to(torch.float32)
    packed3 = pack_weights(w3)
    y = trix_forward(x, packed3, scales, gate, out_f, num_tiles)
    # reference
    ref = (x @ w3.t()) * scales
    out_per_tile = out_f // num_tiles
    for t in range(num_tiles):
        ref[:, t * out_per_tile : (t + 1) * out_per_tile] *= gate[:, t : t + 1]
    if not torch.allclose(y.cpu(), ref.cpu(), atol=1e-5):
        print("FAIL: trix_forward != reference")
        print("Next: run `python -m pytest -q tests/test_kernel_reference_harness.py`")
        return 2

    print("PASS")
    return 0


def _bench(outdir: str, device: str, include_slow: bool) -> int:
    root = _repo_root_from_here()
    if root is None:
        print("Bench requires repo checkout (experiments/ not found).")
        return 2
    script = root / "experiments" / "benchmarks" / "benchmark_suite_v1.py"
    if not script.exists():
        print(f"Missing benchmark script: {script}")
        return 2

    cmd = [sys.executable, str(script), "--outdir", outdir, "--device", device]
    if include_slow:
        cmd.append("--include-slow")
    r = subprocess.run(cmd)
    code = int(r.returncode)

    suite_path = Path(outdir) / "suite_v1.json"
    if suite_path.exists():
        try:
            d = json.loads(suite_path.read_text(encoding="utf-8"))
            ok = bool(d.get("ok"))
            names = [b.get("name") for b in d.get("benchmarks", [])]
            print(f"suite_ok={ok} benchmarks={names}")
        except Exception:
            pass
    return code


def _export_bundle(args: argparse.Namespace) -> int:
    from trix.nn.bundle import export_address_bundle, validate_compiled_semantics
    from trix.nn import SparseLookupFFNv2
    from trix.nn.compiled_dispatch import CompiledDispatch

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("ffn_type") not in (None, "SparseLookupFFNv2"):
        print("Only ffn_type=SparseLookupFFNv2 is supported")
        return 2

    ffn = SparseLookupFFNv2(
        d_model=int(cfg["d_model"]),
        num_tiles=int(cfg["num_tiles"]),
        tiles_per_cluster=int(cfg["tiles_per_cluster"]),
        grid_size=int(cfg.get("grid_size", 16)),
        dropout=0.0,
        use_score_calibration=bool(cfg.get("use_score_calibration", False)),
        routing_backend=str(cfg.get("routing_backend", "hierarchical_dot")),
        ternary_weight=float(cfg.get("ternary_weight", 0.0)),
        sparsity_weight=float(cfg.get("sparsity_weight", 0.0)),
        diversity_weight=float(cfg.get("diversity_weight", 0.0)),
    )

    state = torch.load(args.state_dict, map_location="cpu")
    ffn.load_state_dict(state)

    compiled = None
    if args.compile_stable:
        compiled = CompiledDispatch(ffn)
        compiled.compile_stable(
            threshold=float(args.threshold),
            min_confidence=float(args.min_confidence),
            num_classes=int(args.num_classes) if args.num_classes is not None else None,
        )

    validation = None
    if compiled is not None and args.validate:
        validation = validate_compiled_semantics(
            ffn=ffn, compiled=compiled, samples=64, seed=0
        )

    b = export_address_bundle(
        ffn=ffn,
        outdir=args.outdir,
        compiled=compiled,
        include_state_dict=bool(args.include_state),
        validation=validation,
        extra_meta={"exported_by": "trix export-bundle"},
    )
    print(str(b.outdir))
    return 0


def _load_bundle(args: argparse.Namespace) -> int:
    from trix.nn.bundle import load_address_bundle, validate_compiled_semantics
    from trix.nn.integrity import check_bundle_compatibility, verify_manifest

    try:
        bundle, ffn, compiled = load_address_bundle(
            outdir=args.outdir, device=args.device
        )
    except Exception as e:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(e)}, indent=2, sort_keys=True))
        else:
            print(f"FAIL: {e}")
        return 2
    out: Dict[str, Any] = {
        "bundle": str(bundle.outdir),
        "ffn": bundle.config.get("ffn_type"),
        "routing_backend": bundle.config.get("routing_backend"),
        "has_state_dict": bool(
            bundle.state_dict_path and bundle.state_dict_path.exists()
        ),
        "has_dispatch": compiled is not None,
    }

    if args.validate:
        crep = check_bundle_compatibility(Path(args.outdir))
        out["compatible"] = bool(crep.compatible)
        out["compat_warnings"] = crep.warnings
        out["compat_errors"] = crep.errors
        if crep.errors:
            if getattr(args, "json", False):
                out["ok"] = False
                print(json.dumps(out, indent=2, sort_keys=True))
            else:
                _print_kv("bundle", out["bundle"])
                _print_kv("ffn", out["ffn"])
                _print_kv("routing_backend", out["routing_backend"])
                _print_kv("has_state_dict", out["has_state_dict"])
                _print_kv("has_dispatch", out["has_dispatch"])
                _print_kv("compatible", out["compatible"])
                if crep.warnings:
                    _print_kv("compat_warnings", crep.warnings)
                _print_kv("compat_errors", crep.errors)
                print("FAIL")
            return 2

        mpath = Path(args.outdir) / "manifest.json"
        if mpath.exists():
            mok, merrs = verify_manifest(Path(args.outdir))
            out["manifest_present"] = True
            out["manifest_ok"] = bool(mok)
            out["manifest_errors"] = merrs
            if not mok:
                if getattr(args, "json", False):
                    out["ok"] = False
                    print(json.dumps(out, indent=2, sort_keys=True))
                else:
                    _print_kv("bundle", out["bundle"])
                    _print_kv("ffn", out["ffn"])
                    _print_kv("routing_backend", out["routing_backend"])
                    _print_kv("has_state_dict", out["has_state_dict"])
                    _print_kv("has_dispatch", out["has_dispatch"])
                    _print_kv("compatible", out["compatible"])
                    if crep.warnings:
                        _print_kv("compat_warnings", crep.warnings)
                    _print_kv("manifest_ok", mok)
                    if merrs:
                        _print_kv("manifest_errors", merrs)
                    print("FAIL")
                return 2
        else:
            out["manifest_present"] = False

    if args.validate and compiled is not None:
        rep = validate_compiled_semantics(
            ffn=ffn, compiled=compiled, samples=64, seed=0
        )
        out["max_abs_err_compiled_vs_forced"] = rep["max_abs_err_compiled_vs_forced"]

    if getattr(args, "json", False):
        out["ok"] = True
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    _print_kv("bundle", out["bundle"])
    _print_kv("ffn", out["ffn"])
    _print_kv("routing_backend", out["routing_backend"])
    _print_kv("has_state_dict", out["has_state_dict"])
    _print_kv("has_dispatch", out["has_dispatch"])
    if args.validate:
        _print_kv("compatible", out.get("compatible"))
        if out.get("compat_warnings"):
            _print_kv("compat_warnings", out.get("compat_warnings"))
        if out.get("manifest_present"):
            _print_kv("manifest_ok", out.get("manifest_ok"))
            if out.get("manifest_errors"):
                _print_kv("manifest_errors", out.get("manifest_errors"))
    if "max_abs_err_compiled_vs_forced" in out:
        _print_kv(
            "max_abs_err_compiled_vs_forced", out["max_abs_err_compiled_vs_forced"]
        )
    print("OK")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="trix")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="environment + correctness self-check")

    pb = sub.add_parser("bench", help="run benchmark suite v1")
    pb.add_argument("--outdir", default="results/benchmarks_v1")
    pb.add_argument("--device", default="cpu")
    pb.add_argument("--include-slow", action="store_true")

    pe = sub.add_parser("export-bundle", help="export address bundle from a state_dict")
    pe.add_argument("--config", required=True, help="JSON config describing the FFN")
    pe.add_argument(
        "--state-dict", required=True, dest="state_dict", help="path to state_dict.pt"
    )
    pe.add_argument("--outdir", required=True)
    pe.add_argument("--include-state", action="store_true")
    pe.add_argument("--compile-stable", action="store_true")
    pe.add_argument("--threshold", type=float, default=0.3)
    pe.add_argument("--min-confidence", type=float, default=0.0)
    pe.add_argument("--num-classes", type=int)
    pe.add_argument("--validate", action="store_true")

    pl = sub.add_parser("load-bundle", help="load and optionally validate a bundle")
    pl.add_argument("--outdir", required=True)
    pl.add_argument("--device", default=None)
    pl.add_argument("--validate", action="store_true")
    pl.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    pbun = sub.add_parser("bundle", help="bundle integrity utilities")
    bun = pbun.add_subparsers(dest="bundle_cmd", required=True)
    bman = bun.add_parser("manifest", help="generate manifest.json")
    bman.add_argument("--outdir", required=True)
    bman.add_argument("--json", action="store_true", help="emit manifest JSON")
    bver = bun.add_parser("verify", help="verify manifest.json")
    bver.add_argument("--outdir", required=True)
    bver.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    pd = sub.add_parser("drift", help="drift policy checks")
    dsub = pd.add_subparsers(dest="drift_cmd", required=True)
    dchk = dsub.add_parser("check", help="check policy against suite json")
    dchk.add_argument("--suite", required=True, help="path to suite_v1.json")
    dchk.add_argument("--policy", required=True, help="path to drift policy json")
    dchk.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    ppol = dsub.add_parser("policy", help="drift policy utilities")
    ppolsub = ppol.add_subparsers(dest="policy_cmd", required=True)
    pinit = ppolsub.add_parser("init", help="print a starter drift policy json")
    pinit.add_argument(
        "--out",
        default="-",
        help="output path or '-' for stdout (default '-')",
    )
    pinit.add_argument(
        "--force",
        action="store_true",
        help="overwrite --out if it exists",
    )

    args = p.parse_args(argv)

    if args.cmd == "doctor":
        return _doctor()
    if args.cmd == "bench":
        return _bench(args.outdir, args.device, bool(args.include_slow))
    if args.cmd == "export-bundle":
        return _export_bundle(args)
    if args.cmd == "load-bundle":
        return _load_bundle(args)

    if args.cmd == "bundle":
        from trix.nn.integrity import (
            generate_manifest,
            verify_manifest,
            check_bundle_compatibility,
        )

        outdir = Path(args.outdir)
        if args.bundle_cmd == "manifest":
            try:
                m = generate_manifest(outdir)
            except Exception as e:
                if getattr(args, "json", False):
                    print(
                        json.dumps(
                            {"ok": False, "error": str(e)}, indent=2, sort_keys=True
                        )
                    )
                else:
                    print(f"FAIL: {e}")
                return 2
            if getattr(args, "json", False):
                print(json.dumps(m, indent=2, sort_keys=True))
            else:
                print("OK")
            return 0
        if args.bundle_cmd == "verify":
            rep = check_bundle_compatibility(outdir)
            ok, errs = verify_manifest(outdir)
            if not ok:
                if getattr(args, "json", False):
                    payload = {
                        "ok": False,
                        "compatible": bool(rep.compatible),
                        "compat_warnings": rep.warnings,
                        "compat_errors": rep.errors,
                        "errors": errs,
                    }
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print("FAIL: manifest verification failed")
                    if rep.warnings:
                        print("warnings:")
                        for w in rep.warnings:
                            print(f"- {w}")
                    for e in errs:
                        print(f"- {e}")
                return 2
            if getattr(args, "json", False):
                payload = {
                    "ok": True,
                    "compatible": bool(rep.compatible),
                    "compat_warnings": rep.warnings,
                }
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print("OK")
            return 0

    if args.cmd == "drift":
        from trix.nn.integrity import load_drift_policy, evaluate_drift_policy_on_suite

        if args.drift_cmd == "check":
            pol = load_drift_policy(Path(args.policy))
            ok, rep = evaluate_drift_policy_on_suite(Path(args.suite), pol)
            # Default output is JSON already; --json guarantees no extra output.
            print(json.dumps(rep, indent=2, sort_keys=True))
            return 0 if ok else 2

        if args.drift_cmd == "policy":
            if args.policy_cmd == "init":
                starter = {
                    "policy_version": 1,
                    "drift_threshold": 0.2,
                    "max_churn": 0.1,
                    "max_near_tie_rate": 0.3,
                    "min_margin_mean": 0.001,
                    "on_violation": "fallback",
                }

                outp = str(args.out)
                if outp == "-":
                    print(json.dumps(starter, indent=2, sort_keys=True))
                    return 0

                p = Path(outp)
                if p.exists() and not bool(args.force):
                    print(f"FAIL: exists: {p}")
                    return 2
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(
                    json.dumps(starter, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(str(p))
                return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
