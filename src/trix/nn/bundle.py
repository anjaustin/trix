"""Address bundle utilities.

Mesa 14 goal: make routing artifacts portable and validated.

Bundle contents (directory):
- bundle.json: metadata + config
- compressed_signatures.json: CompressedSignatures export
- dispatch_table.json: CompiledDispatch export (optional)
- validation.json: validation report (optional)
- state_dict.pt: model state_dict (optional)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .compiled_dispatch import CompiledDispatch
from .sparse_lookup_v2 import SparseLookupFFNv2
from .xor_superposition import CompressedSignatures
from .integrity import generate_manifest


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class AddressBundle:
    outdir: Path
    meta: Dict[str, Any]
    compressed_signatures: Dict[str, Any]
    dispatch_table: Optional[Dict[str, Any]]
    validation: Optional[Dict[str, Any]]
    config: Dict[str, Any]
    state_dict_path: Optional[Path]


def export_address_bundle(
    *,
    ffn: SparseLookupFFNv2,
    outdir: str | Path,
    compiled: Optional[CompiledDispatch] = None,
    include_state_dict: bool = False,
    validation: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> AddressBundle:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    ffn.compress_signatures()
    comp_sig = ffn.export_compressed_signatures()

    dispatch = compiled.export_dispatch_table() if compiled is not None else None

    state_path = None
    if include_state_dict:
        state_path = out / "state_dict.pt"
        torch.save(ffn.state_dict(), state_path)

    config = {
        "ffn_type": "SparseLookupFFNv2",
        "d_model": int(ffn.d_model),
        "num_tiles": int(ffn.num_tiles),
        "tiles_per_cluster": int(ffn.tiles_per_cluster),
        "grid_size": int(getattr(ffn, "grid_size", 16)),
        "use_score_calibration": bool(getattr(ffn, "use_score_calibration", False)),
        "routing_backend": str(getattr(ffn, "routing_backend", "hierarchical_dot")),
        "ternary_weight": float(getattr(ffn, "ternary_weight", 0.0)),
        "sparsity_weight": float(getattr(ffn, "sparsity_weight", 0.0)),
        "diversity_weight": float(getattr(ffn, "diversity_weight", 0.0)),
    }

    meta = {
        "schema_version": 1,
        "created_at": int(time.time()),
        "trix_version": getattr(__import__("trix"), "__version__", "unknown"),
    }
    if extra_meta:
        meta.update(extra_meta)

    _write_json(out / "compressed_signatures.json", comp_sig)
    if dispatch is not None:
        _write_json(out / "dispatch_table.json", dispatch)
    if validation is not None:
        _write_json(out / "validation.json", validation)

    bundle_json = {
        "meta": meta,
        "config": config,
        "files": {
            "compressed_signatures": "compressed_signatures.json",
            "dispatch_table": "dispatch_table.json" if dispatch is not None else None,
            "validation": "validation.json" if validation is not None else None,
            "state_dict": "state_dict.pt" if state_path is not None else None,
        },
    }
    _write_json(out / "bundle.json", bundle_json)

    # Mesa 15: always generate a manifest for tamper detection.
    try:
        generate_manifest(out)
    except Exception:
        # Do not fail export for manifest issues.
        pass

    return AddressBundle(
        outdir=out,
        meta=meta,
        compressed_signatures=comp_sig,
        dispatch_table=dispatch,
        validation=validation,
        config=config,
        state_dict_path=state_path,
    )


def load_address_bundle(
    *,
    outdir: str | Path,
    device: Optional[str] = None,
) -> tuple[AddressBundle, SparseLookupFFNv2, Optional[CompiledDispatch]]:
    root = Path(outdir)
    b = _read_json(root / "bundle.json")
    cfg = b["config"]

    if cfg.get("ffn_type") != "SparseLookupFFNv2":
        raise ValueError("only SparseLookupFFNv2 bundles are supported in v1")

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

    dev = torch.device(device) if device else None
    if dev is not None:
        ffn = ffn.to(dev)

    files = b.get("files", {})
    state_rel = files.get("state_dict")
    state_path = (root / state_rel) if state_rel else None
    if state_path is not None and state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        ffn.load_state_dict(state)

    comp_sig = _read_json(root / files["compressed_signatures"])
    ffn.import_compressed_signatures(comp_sig)

    compiled = None
    dispatch_rel = files.get("dispatch_table")
    if dispatch_rel:
        dispatch = _read_json(root / dispatch_rel)
        compiled = CompiledDispatch(ffn)
        compiled.import_dispatch_table(dispatch)

    validation = None
    val_rel = files.get("validation")
    if val_rel:
        validation = _read_json(root / val_rel)

    meta = b.get("meta", {})

    bundle = AddressBundle(
        outdir=root,
        meta=meta,
        compressed_signatures=comp_sig,
        dispatch_table=_read_json(root / dispatch_rel) if dispatch_rel else None,
        validation=validation,
        config=cfg,
        state_dict_path=state_path,
    )
    return bundle, ffn, compiled


@torch.no_grad()
def validate_compiled_semantics(
    *,
    ffn: SparseLookupFFNv2,
    compiled: CompiledDispatch,
    samples: int = 64,
    seed: int = 0,
) -> Dict[str, Any]:
    """Validate compiled execution equals forced-tile execution."""
    torch.manual_seed(seed)
    ffn.eval()
    compiled.reset_stats()

    d = int(ffn.d_model)
    x = torch.randn(4, max(1, samples // 4), d, device=next(ffn.parameters()).device)

    table = compiled.export_dispatch_table()
    entries = table.get("entries", {})
    compiled_classes = []
    max_err = 0.0

    for class_id_s, e in entries.items():
        class_id = int(class_id_s)
        tile_idx = int(e["tile_idx"])
        compiled_classes.append(class_id)

        out_c, info_c, _aux_c = compiled.forward(x, class_hint=class_id, confidence=1.0)
        out_f, info_f, _aux_f = ffn.forward_forced_tile(x, tile_idx=tile_idx)
        err = float((out_c - out_f).abs().max().item())
        max_err = max(max_err, err)

        if info_c.get("compiled") is True:
            if not torch.equal(info_c["tile_idx"], info_f["tile_idx"]):
                raise AssertionError("compiled tile_idx mismatch")

    return {
        "compiled_classes": sorted(set(compiled_classes)),
        "max_abs_err_compiled_vs_forced": float(max_err),
        "compiled_stats": compiled.get_stats(),
    }
