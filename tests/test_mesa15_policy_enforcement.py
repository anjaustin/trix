from pathlib import Path

import pytest
import torch


def test_policy_denies_compiled_tile_and_falls_back(tmp_path: Path):
    from trix.nn import SparseLookupFFNv2
    from trix.nn.compiled_dispatch import CompiledDispatch
    from trix.nn.policy import AddressPolicyV1

    torch.manual_seed(0)
    ffn = SparseLookupFFNv2(
        d_model=16,
        num_tiles=4,
        tiles_per_cluster=2,
        dropout=0.0,
        use_score_calibration=False,
    )

    compiled = CompiledDispatch(ffn)
    compiled.compile(class_id=0, tile_idx=0, min_confidence=0.0)

    # Deny the compiled tile so we must fall back to dynamic routing.
    policy = AddressPolicyV1.from_lists(deny_tiles=[0], on_violation="fallback")

    x = torch.randn(2, 3, 16)
    out, routing_info, aux = compiled.forward(
        x, class_hint=0, confidence=1.0, tile_policy=policy
    )

    assert out.shape == x.shape
    assert isinstance(aux, dict)
    assert routing_info.get("compiled") is False
    assert routing_info.get("policy_violation") is True
    assert routing_info.get("policy_reason") == "compiled_tile_denied"

    tiles = routing_info["tile_idx"].detach().cpu().numpy()
    assert (tiles != 0).all()
    assert routing_info.get("policy_fallback_applied") is True
    assert routing_info.get("policy_num_violations", 0) >= 1


def test_policy_fail_raises_when_disallowed(tmp_path: Path):
    from trix.nn import SparseLookupFFNv2
    from trix.nn.policy import AddressPolicyV1

    ffn = SparseLookupFFNv2(
        d_model=8,
        num_tiles=2,
        tiles_per_cluster=1,
        dropout=0.0,
        use_score_calibration=False,
    )
    x = torch.randn(1, 2, 8)

    policy = AddressPolicyV1.from_lists(deny_tiles=[0, 1], on_violation="fail")
    with pytest.raises(RuntimeError):
        ffn(x, tile_policy=policy)
