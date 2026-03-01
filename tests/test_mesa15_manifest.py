from pathlib import Path

import pytest
import torch


def test_manifest_generate_and_verify_roundtrip(tmp_path: Path):
    from trix.nn import SparseLookupFFNv2
    from trix.nn.bundle import export_address_bundle
    from trix.nn.integrity import verify_manifest

    torch.manual_seed(0)
    ffn = SparseLookupFFNv2(
        d_model=16,
        num_tiles=4,
        tiles_per_cluster=2,
        dropout=0.0,
        use_score_calibration=False,
    )
    outdir = tmp_path / "b"
    export_address_bundle(ffn=ffn, outdir=outdir, include_state_dict=True)

    ok, errs = verify_manifest(outdir)
    assert ok
    assert errs == []


def test_manifest_detects_tamper(tmp_path: Path):
    from trix.nn import SparseLookupFFNv2
    from trix.nn.bundle import export_address_bundle
    from trix.nn.integrity import verify_manifest

    ffn = SparseLookupFFNv2(
        d_model=16,
        num_tiles=4,
        tiles_per_cluster=2,
        dropout=0.0,
        use_score_calibration=False,
    )
    outdir = tmp_path / "b"
    export_address_bundle(ffn=ffn, outdir=outdir, include_state_dict=True)

    # Tamper with compressed signatures
    p = outdir / "compressed_signatures.json"
    txt = p.read_text(encoding="utf-8")
    p.write_text(txt + "\n", encoding="utf-8")

    ok, errs = verify_manifest(outdir)
    assert not ok
    assert any("hash mismatch" in e for e in errs)
