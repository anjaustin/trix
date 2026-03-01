import torch


def test_compiled_dispatch_uses_forced_tile_path():
    from trix.nn import SparseLookupFFNv2
    from trix.nn.compiled_dispatch import CompiledDispatch

    torch.manual_seed(0)
    ffn = SparseLookupFFNv2(d_model=32, num_tiles=8, tiles_per_cluster=4, dropout=0.0)
    ffn.eval()

    compiler = CompiledDispatch(ffn)
    compiler.compile(class_id=0, tile_idx=3, min_confidence=0.0)

    x = torch.randn(2, 4, 32)
    out_c, info_c, aux_c = compiler.forward(x, class_hint=0, confidence=1.0)
    out_f, info_f, aux_f = ffn.forward_forced_tile(x, tile_idx=3)

    assert info_c["compiled"] is True
    assert info_c["compiled_class"] == 0
    assert info_c["tile_idx"].shape == (2, 4)

    # Compiled output should match forced-tile output (same semantics).
    assert torch.allclose(out_c, out_f)
    assert torch.equal(info_f["tile_idx"], info_c["tile_idx"])
    assert float(aux_c["total_aux"].item()) == float(aux_f["total_aux"].item())
