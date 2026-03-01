import torch


def test_policy_fallback_reroutes_under_flat_popcount():
    from trix.nn import SparseLookupFFNv2
    from trix.nn.policy import AddressPolicyV1

    d = 16
    ffn = SparseLookupFFNv2(
        d_model=d,
        num_tiles=2,
        tiles_per_cluster=1,
        dropout=0.0,
        use_score_calibration=False,
        routing_backend="flat_popcount",
    )

    # Make deterministic ternary signatures.
    s0 = torch.ones(d)
    s1 = -torch.ones(d)
    with torch.no_grad():
        ffn.signatures_raw[0].copy_(s0)
        ffn.signatures_raw[1].copy_(s1)

    # x matches tile 0 exactly under sign(), so routing would pick tile 0.
    x = torch.ones(1, 4, d)

    policy = AddressPolicyV1.from_lists(deny_tiles=[0], on_violation="fallback")
    _out, info, _aux = ffn(x, tile_policy=policy)

    tile_idx = info["tile_idx"].detach().cpu()
    assert (tile_idx == 1).all()
    assert info.get("policy_fallback_applied") is True
    assert info.get("policy_num_violations", 0) > 0
