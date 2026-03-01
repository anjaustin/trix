import torch


def test_sparse_lookup_v2_compress_export_import_roundtrip():
    from trix.nn import SparseLookupFFNv2

    torch.manual_seed(0)
    ffn = SparseLookupFFNv2(
        d_model=33,
        num_tiles=8,
        tiles_per_cluster=4,
        dropout=0.0,
        use_score_calibration=False,
        routing_backend="flat_popcount",
    )

    # Build compressed signatures from current state.
    ffn.compress_signatures()
    stats = ffn.get_signature_compression_stats()
    assert stats is not None
    assert stats["num_signatures"] == 8
    assert stats["dim"] == 33

    data = ffn.export_compressed_signatures()

    # Import into a fresh module and ensure signatures match.
    ffn2 = SparseLookupFFNv2(
        d_model=33,
        num_tiles=8,
        tiles_per_cluster=4,
        dropout=0.0,
        use_score_calibration=False,
        routing_backend="flat_popcount",
    )
    ffn2.import_compressed_signatures(data)
    assert torch.equal(ffn.signatures.cpu(), ffn2.signatures.cpu())

    # Routing decisions match on ternary inputs.
    x = torch.sign(torch.randn(64, 33))
    idx1, _ = ffn.route(x, ffn.signatures, backend="flat_popcount", return_scores=False)
    idx2, _ = ffn2.route(
        x, ffn2.signatures, backend="flat_popcount", return_scores=False
    )
    assert torch.equal(idx1, idx2)
