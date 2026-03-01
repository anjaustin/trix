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


def test_cli_bundle_verify_json_roundtrip(tmp_path: Path):
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
    export_address_bundle(ffn=ffn, outdir=bdir, include_state_dict=True)

    r = _run_cli(["bundle", "verify", "--outdir", str(bdir), "--json"])
    assert r.returncode == 0, (r.stdout, r.stderr)
    payload = json.loads(r.stdout)
    assert payload["ok"] is True

    # Tamper a file and ensure verify fails.
    p = bdir / "compressed_signatures.json"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    r2 = _run_cli(["bundle", "verify", "--outdir", str(bdir), "--json"])
    assert r2.returncode == 2
    payload2 = json.loads(r2.stdout)
    assert payload2["ok"] is False
    assert any("hash mismatch" in e for e in payload2.get("errors", []))


def test_cli_load_bundle_validate_json(tmp_path: Path):
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
    export_address_bundle(ffn=ffn, outdir=bdir, include_state_dict=True)

    r = _run_cli(["load-bundle", "--outdir", str(bdir), "--validate", "--json"])
    assert r.returncode == 0, (r.stdout, r.stderr)
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    assert payload["compatible"] is True
    assert payload["manifest_present"] is True
    assert payload["manifest_ok"] is True


def test_cli_drift_policy_init(tmp_path: Path):
    outp = tmp_path / "drift_policy.json"

    r = _run_cli(["drift", "policy", "init", "--out", str(outp)])
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert outp.exists()
    d = json.loads(outp.read_text(encoding="utf-8"))
    assert d["policy_version"] == 1
    assert "max_churn" in d

    # Refuse overwrite without --force.
    r2 = _run_cli(["drift", "policy", "init", "--out", str(outp)])
    assert r2.returncode == 2

    # Overwrite with --force.
    r3 = _run_cli(["drift", "policy", "init", "--out", str(outp), "--force"])
    assert r3.returncode == 0
