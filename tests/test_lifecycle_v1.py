import json
from pathlib import Path

import torch


def test_lifecycle_observe_and_jsonl(tmp_path: Path):
    from trix.nn import SparseLookupFFNv2, RoutingLifecycleV1

    torch.manual_seed(0)
    ffn = SparseLookupFFNv2(d_model=32, num_tiles=8, tiles_per_cluster=4, dropout=0.0)
    ffn.eval()

    lc = RoutingLifecycleV1(ffn)
    x = torch.randn(2, 4, 32)

    out, routing_info, aux, record = lc.observe(
        x, jsonl_path=str(tmp_path / "telemetry.jsonl")
    )
    assert out.shape == x.shape
    assert "tile_idx" in routing_info
    assert isinstance(aux, dict)
    assert record["schema_version"] == 1
    assert record["event"] == "routing"

    lines = (
        (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(lines) == 1
    r = json.loads(lines[0])
    assert r["event"] == "routing"
    assert "margin_mean" in r


def test_lifecycle_edit_and_undo():
    from trix.nn import SparseLookupFFNv2, RoutingLifecycleV1

    torch.manual_seed(0)
    ffn = SparseLookupFFNv2(d_model=16, num_tiles=4, tiles_per_cluster=2, dropout=0.0)
    lc = RoutingLifecycleV1(ffn)

    tile_idx = 0
    before = ffn.signatures_raw[tile_idx].detach().clone()
    sig = torch.zeros(16)
    sig[:4] = 1.0

    op = lc.edit_insert_signature(
        tile_idx=tile_idx, signature=sig, freeze=True, tag="t"
    )
    assert ffn.is_frozen(tile_idx)
    assert not torch.allclose(ffn.signatures_raw[tile_idx].detach(), before)

    lc.undo(op)
    assert torch.allclose(ffn.signatures_raw[tile_idx].detach(), before)


def test_lifecycle_compile_export_import_roundtrip():
    from trix.nn import SparseLookupFFNv2, CompiledDispatch, RoutingLifecycleV1

    torch.manual_seed(0)
    ffn = SparseLookupFFNv2(d_model=32, num_tiles=8, tiles_per_cluster=4, dropout=0.0)
    ffn.train()

    # Populate claim_matrix with a tiny synthetic run.
    x = torch.randn(4, 8, 32)
    labels = torch.randint(0, 8, (4, 8))
    _out, _info, _aux = ffn(x, labels=labels)

    comp = CompiledDispatch(ffn)
    lc = RoutingLifecycleV1(ffn, compiled=comp)

    compiled = lc.compile_stable(threshold=0.0, min_confidence=0.0, num_classes=8)
    assert isinstance(compiled, dict)

    exported = lc.export_dispatch_table()
    assert "entries" in exported

    comp2 = CompiledDispatch(ffn)
    lc2 = RoutingLifecycleV1(ffn, compiled=comp2)
    lc2.import_dispatch_table(exported)
    assert comp2.export_dispatch_table() == exported


def test_lifecycle_stability_probe_jsonl(tmp_path: Path):
    from trix.nn import SparseLookupFFNv2, RoutingLifecycleV1

    torch.manual_seed(0)
    ffn = SparseLookupFFNv2(d_model=32, num_tiles=8, tiles_per_cluster=4, dropout=0.0)
    ffn.eval()
    lc = RoutingLifecycleV1(ffn)

    x = torch.randn(2, 4, 32)
    rec = lc.stability_probe(
        x, flip_prob=0.05, seed=123, jsonl_path=str(tmp_path / "s.jsonl")
    )
    assert rec["event"] == "stability"
    assert 0.0 <= rec["churn"] <= 1.0

    lines = (tmp_path / "s.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    r = json.loads(lines[0])
    assert r["event"] == "stability"
