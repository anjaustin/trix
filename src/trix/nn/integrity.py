"""Mesa 15: bundle integrity, compatibility, and drift policy utilities."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compute_fingerprints(
    bundle_dir: Path, bundle_json: Dict[str, Any]
) -> Dict[str, Optional[str]]:
    files = bundle_json.get("files", {})
    cfg = bundle_json.get("config", {})

    compressed = _read_json(bundle_dir / files["compressed_signatures"])

    dispatch = None
    if files.get("dispatch_table"):
        dispatch = _read_json(bundle_dir / files["dispatch_table"])

    validation = None
    if files.get("validation"):
        validation = _read_json(bundle_dir / files["validation"])

    return {
        "config_fingerprint": sha256_json(cfg),
        "address_plane_fingerprint": sha256_json(compressed),
        "contract_fingerprint": sha256_json(dispatch) if dispatch is not None else None,
        "validation_fingerprint": sha256_json(validation)
        if validation is not None
        else None,
    }


@dataclass
class CompatibilityReport:
    compatible: bool
    warnings: list[str]
    errors: list[str]


def check_bundle_compatibility(bundle_dir: Path) -> CompatibilityReport:
    warnings: list[str] = []
    errors: list[str] = []

    bundle_json_path = bundle_dir / "bundle.json"
    if not bundle_json_path.exists():
        errors.append("missing bundle.json")
        return CompatibilityReport(False, warnings, errors)

    b = _read_json(bundle_json_path)
    meta = b.get("meta", {})
    schema = int(meta.get("schema_version", 0))
    if schema != 1:
        errors.append(f"unsupported bundle schema_version={schema}")

    files = b.get("files", {})
    required = ["compressed_signatures"]
    for k in required:
        rel = files.get(k)
        if not rel:
            errors.append(f"missing required file entry: files.{k}")
        else:
            p = bundle_dir / rel
            if not p.exists():
                errors.append(f"missing required file: {rel}")

    cfg = b.get("config", {})
    if cfg.get("ffn_type") != "SparseLookupFFNv2":
        errors.append("only SparseLookupFFNv2 bundles are supported in v1")

    backend = cfg.get("routing_backend")
    if backend not in ("hierarchical_dot", "flat_popcount"):
        warnings.append(f"unknown routing_backend={backend!r}")

    trix_version = meta.get("trix_version")
    if trix_version is None:
        warnings.append("bundle meta missing trix_version")

    compatible = len(errors) == 0
    return CompatibilityReport(compatible, warnings, errors)


def generate_manifest(bundle_dir: Path, *, refresh: bool = True) -> Dict[str, Any]:
    """Generate manifest.json with sha256 for bundle files."""

    bundle_dir = Path(bundle_dir)
    rep = check_bundle_compatibility(bundle_dir)
    if not rep.compatible:
        raise ValueError(
            "cannot generate manifest for incompatible bundle: " + "; ".join(rep.errors)
        )

    b = _read_json(bundle_dir / "bundle.json")
    files = b.get("files", {})
    manifest_files = {
        "bundle.json": "bundle.json",
        "compressed_signatures.json": files.get("compressed_signatures"),
        "dispatch_table.json": files.get("dispatch_table"),
        "validation.json": files.get("validation"),
        "state_dict.pt": files.get("state_dict"),
    }

    file_hashes: Dict[str, str] = {}
    for logical, rel in manifest_files.items():
        if rel is None:
            continue
        p = bundle_dir / rel
        if p.exists():
            file_hashes[rel] = sha256_file(p)

    fps = _compute_fingerprints(bundle_dir, b)

    out = {
        "manifest_version": 1,
        "bundle_schema_version": int(b.get("meta", {}).get("schema_version", 0)),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "trix_version": b.get("meta", {}).get("trix_version"),
        "file_hashes": file_hashes,
        **fps,
    }

    if refresh:
        _write_json(bundle_dir / "manifest.json", out)
    return out


def verify_manifest(bundle_dir: Path) -> Tuple[bool, list[str]]:
    bundle_dir = Path(bundle_dir)
    errs: list[str] = []

    rep = check_bundle_compatibility(bundle_dir)
    if not rep.compatible:
        errs.extend([f"incompatible bundle: {e}" for e in rep.errors])
        return False, errs

    mpath = bundle_dir / "manifest.json"
    if not mpath.exists():
        return False, ["missing manifest.json"]

    m = _read_json(mpath)
    if int(m.get("manifest_version", 0)) != 1:
        errs.append("unsupported manifest_version")

    # Ensure schema version is consistent.
    try:
        b = _read_json(bundle_dir / "bundle.json")
        schema = int(b.get("meta", {}).get("schema_version", 0))
        if int(m.get("bundle_schema_version", -1)) != schema:
            errs.append("bundle_schema_version mismatch")
    except Exception:
        errs.append("failed to read bundle.json for schema check")

    expected = m.get("file_hashes", {})
    if not isinstance(expected, dict) or not expected:
        errs.append("manifest missing file_hashes")
        return False, errs

    for rel, hexh in expected.items():
        p = bundle_dir / rel
        if not p.exists():
            errs.append(f"missing file: {rel}")
            continue
        got = sha256_file(p)
        if got != hexh:
            errs.append(f"hash mismatch: {rel}")

    # Recompute fingerprints for stronger semantic checks (without re-hashing files).
    try:
        b = _read_json(bundle_dir / "bundle.json")
        fps = _compute_fingerprints(bundle_dir, b)
        for k, v in fps.items():
            if m.get(k) != v:
                errs.append(f"fingerprint mismatch: {k}")
    except Exception as e:
        errs.append(f"failed to recompute fingerprints: {e}")

    ok = len(errs) == 0
    return ok, errs


@dataclass
class DriftPolicy:
    policy_version: int = 1
    drift_threshold: float = 0.2
    max_churn: float = 0.10
    max_near_tie_rate: float = 0.30
    min_margin_mean: float = 1e-3
    on_violation: str = "fallback"  # fallback|fail|recompile


def load_drift_policy(path: Path) -> DriftPolicy:
    d = _read_json(Path(path))
    pv = int(d.get("policy_version", 0))
    if pv != 1:
        raise ValueError("unsupported drift policy version")
    on = str(d.get("on_violation", "fallback"))
    if on not in {"fallback", "fail", "recompile"}:
        raise ValueError("invalid on_violation")
    return DriftPolicy(
        policy_version=pv,
        drift_threshold=float(d.get("drift_threshold", 0.2)),
        max_churn=float(d.get("max_churn", 0.10)),
        max_near_tie_rate=float(d.get("max_near_tie_rate", 0.30)),
        min_margin_mean=float(d.get("min_margin_mean", 1e-3)),
        on_violation=on,
    )


def evaluate_drift_policy_on_suite(
    suite_json_path: Path, policy: DriftPolicy
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate drift policy against suite_v1.json drift benchmark outputs."""
    suite = _read_json(Path(suite_json_path))
    drift = None
    for b in suite.get("benchmarks", []):
        if b.get("name") == "drift_under_regularizer_training":
            drift = b
            break
    if drift is None:
        raise ValueError("suite missing drift_under_regularizer_training benchmark")

    metrics = drift.get("metrics", {})
    churn = metrics.get("churn", [])
    if not churn:
        raise ValueError("drift benchmark missing churn")

    drifted_classes = metrics.get("drifted_classes", [])
    compiled_hit_rate = metrics.get("compiled_hit_rate", [])

    compiled_classes = drift.get("compiled", {}).get("compiled_classes", [])
    denom = max(
        1,
        int(len(compiled_classes))
        or int(drift.get("config", {}).get("num_classes", 1)),
    )

    drift_frac = []
    if isinstance(drifted_classes, list) and drifted_classes:
        for step_list in drifted_classes:
            if isinstance(step_list, list):
                drift_frac.append(float(len(step_list)) / float(denom))

    max_churn = float(max(churn))
    max_drift_fraction = float(max(drift_frac)) if drift_frac else None
    min_hit_rate = float(min(compiled_hit_rate)) if compiled_hit_rate else None

    # Optional: parse telemetry jsonl for near-tie + margin signals.
    telemetry_jsonl = drift.get("telemetry", {}).get("jsonl")
    near_tie_max = None
    margin_mean_mean = None
    if telemetry_jsonl:
        try:
            vals_near: list[float] = []
            vals_margin: list[float] = []
            with Path(telemetry_jsonl).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("event") != "routing":
                        continue
                    if "near_tie_rate" in rec:
                        vals_near.append(float(rec["near_tie_rate"]))
                    if "margin_mean" in rec:
                        vals_margin.append(float(rec["margin_mean"]))
            if vals_near:
                near_tie_max = float(max(vals_near))
            if vals_margin:
                margin_mean_mean = float(sum(vals_margin) / len(vals_margin))
        except Exception:
            # Telemetry is optional; do not fail drift evaluation.
            pass

    ok = True
    reasons: list[str] = []
    if max_churn > policy.max_churn:
        ok = False
        reasons.append(f"max_churn {max_churn:.3f} > {policy.max_churn:.3f}")

    if max_drift_fraction is not None and max_drift_fraction > policy.drift_threshold:
        ok = False
        reasons.append(
            f"max_drift_fraction {max_drift_fraction:.3f} > {policy.drift_threshold:.3f}"
        )

    if near_tie_max is not None and near_tie_max > policy.max_near_tie_rate:
        ok = False
        reasons.append(
            f"near_tie_rate_max {near_tie_max:.3f} > {policy.max_near_tie_rate:.3f}"
        )

    if margin_mean_mean is not None and margin_mean_mean < policy.min_margin_mean:
        ok = False
        reasons.append(
            f"margin_mean_mean {margin_mean_mean:.6f} < {policy.min_margin_mean:.6f}"
        )

    report = {
        "policy": {
            "policy_version": policy.policy_version,
            "drift_threshold": policy.drift_threshold,
            "max_churn": policy.max_churn,
            "max_near_tie_rate": policy.max_near_tie_rate,
            "min_margin_mean": policy.min_margin_mean,
            "on_violation": policy.on_violation,
        },
        "measured": {
            "max_churn": max_churn,
            "max_drift_fraction": max_drift_fraction,
            "compiled_hit_rate_min": min_hit_rate,
            "near_tie_rate_max": near_tie_max,
            "margin_mean_mean": margin_mean_mean,
        },
        "ok": ok,
        "reasons": reasons,
    }
    return ok, report
