import torch


def test_address_bundle_export_load_roundtrip(tmp_path):
    from trix.nn import SparseLookupFFNv2
    from trix.nn.compiled_dispatch import CompiledDispatch
    from trix.nn.bundle import (
        export_address_bundle,
        load_address_bundle,
        validate_compiled_semantics,
    )

    torch.manual_seed(0)

    ffn = SparseLookupFFNv2(
        d_model=16,
        num_tiles=4,
        tiles_per_cluster=2,
        dropout=0.0,
        use_score_calibration=False,
        routing_backend="flat_popcount",
    )
    ffn.eval()

    # Make signatures deterministic and compressible.
    for t in range(ffn.num_tiles):
        sig = torch.zeros(16)
        sig[t : t + 2] = 1.0
        ffn.insert_signature(t, sig, freeze=True, tag=f"t{t}")

    comp = CompiledDispatch(ffn)
    comp.compile(class_id=0, tile_idx=2, min_confidence=0.0)
    rep = validate_compiled_semantics(ffn=ffn, compiled=comp, samples=16, seed=0)
    assert rep["max_abs_err_compiled_vs_forced"] == 0.0

    outdir = tmp_path / "bundle"
    export_address_bundle(
        ffn=ffn,
        compiled=comp,
        outdir=outdir,
        include_state_dict=True,
        validation=rep,
        extra_meta={"test": True},
    )

    bundle, ffn2, comp2 = load_address_bundle(outdir=outdir)
    assert bundle.meta["schema_version"] == 1
    assert bundle.config["routing_backend"] == "flat_popcount"
    assert comp2 is not None

    # Signatures should match at the ternary level.
    assert torch.equal(ffn.signatures.cpu(), ffn2.signatures.cpu())

    # Dispatch table should match.
    assert comp.export_dispatch_table() == comp2.export_dispatch_table()

    rep2 = validate_compiled_semantics(ffn=ffn2, compiled=comp2, samples=16, seed=0)
    assert rep2["max_abs_err_compiled_vs_forced"] == 0.0
