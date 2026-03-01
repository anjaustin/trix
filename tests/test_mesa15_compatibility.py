from pathlib import Path

import torch


def test_bundle_compatibility_report(tmp_path: Path):
    from trix.nn import SparseLookupFFNv2
    from trix.nn.bundle import export_address_bundle
    from trix.nn.integrity import check_bundle_compatibility

    ffn = SparseLookupFFNv2(
        d_model=16,
        num_tiles=4,
        tiles_per_cluster=2,
        dropout=0.0,
        use_score_calibration=False,
    )
    outdir = tmp_path / "b"
    export_address_bundle(ffn=ffn, outdir=outdir)

    rep = check_bundle_compatibility(outdir)
    assert rep.compatible
    assert rep.errors == []
