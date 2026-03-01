import json
import subprocess
import sys
from pathlib import Path

import torch


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "trix.cli", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_bundle_verify_json_missing_manifest(tmp_path: Path):
    from trix.nn import SparseLookupFFNv2
    from trix.nn.bundle import export_address_bundle

    ffn = SparseLookupFFNv2(
        d_model=16,
        num_tiles=4,
        tiles_per_cluster=2,
        dropout=0.0,
        use_score_calibration=False,
    )
    bdir = tmp_path / "b"
    export_address_bundle(ffn=ffn, outdir=bdir, include_state_dict=False)
    (bdir / "manifest.json").unlink()

    r = _run_cli(["bundle", "verify", "--outdir", str(bdir), "--json"])
    assert r.returncode == 2
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    assert any("manifest.json" in e for e in payload.get("errors", []))


def test_bundle_verify_json_missing_bundle_json(tmp_path: Path):
    bdir = tmp_path / "empty"
    bdir.mkdir(parents=True, exist_ok=True)
    r = _run_cli(["bundle", "verify", "--outdir", str(bdir), "--json"])
    assert r.returncode == 2
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    assert any("missing bundle.json" in e for e in payload.get("errors", []))


def test_bundle_verify_json_incompatible_schema(tmp_path: Path):
    from trix.nn import SparseLookupFFNv2
    from trix.nn.bundle import export_address_bundle

    torch.manual_seed(0)
    ffn = SparseLookupFFNv2(
        d_model=16,
        num_tiles=4,
        tiles_per_cluster=2,
        dropout=0.0,
        use_score_calibration=False,
    )
    bdir = tmp_path / "b"
    export_address_bundle(ffn=ffn, outdir=bdir, include_state_dict=False)

    bpath = bdir / "bundle.json"
    b = json.loads(bpath.read_text(encoding="utf-8"))
    b["meta"]["schema_version"] = 999
    bpath.write_text(json.dumps(b, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    r = _run_cli(["bundle", "verify", "--outdir", str(bdir), "--json"])
    assert r.returncode == 2
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    assert any("schema_version" in e for e in payload.get("errors", []))


def test_load_bundle_validate_json_missing_bundle_json(tmp_path: Path):
    bdir = tmp_path / "empty"
    bdir.mkdir(parents=True, exist_ok=True)

    r = _run_cli(["load-bundle", "--outdir", str(bdir), "--validate", "--json"])
    assert r.returncode == 2
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    assert "error" in payload


def test_load_bundle_validate_json_incompatible_schema(tmp_path: Path):
    from trix.nn import SparseLookupFFNv2
    from trix.nn.bundle import export_address_bundle

    ffn = SparseLookupFFNv2(
        d_model=16,
        num_tiles=4,
        tiles_per_cluster=2,
        dropout=0.0,
        use_score_calibration=False,
    )
    bdir = tmp_path / "b"
    export_address_bundle(ffn=ffn, outdir=bdir, include_state_dict=False)

    bpath = bdir / "bundle.json"
    b = json.loads(bpath.read_text(encoding="utf-8"))
    b["meta"]["schema_version"] = 999
    bpath.write_text(json.dumps(b, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    r = _run_cli(["load-bundle", "--outdir", str(bdir), "--validate", "--json"])
    assert r.returncode == 2
    payload = json.loads(r.stdout)
    assert payload["ok"] is False
    assert payload.get("compatible") is False
    assert payload.get("compat_errors")
