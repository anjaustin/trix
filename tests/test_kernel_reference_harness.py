import pytest
import torch


def _rand_ternary(rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(0, 3, (rows, cols), generator=g)
    w = torch.zeros((rows, cols), dtype=torch.float32)
    w[r == 0] = -1.0
    w[r == 2] = 1.0
    return w


def _trix_forward_reference(
    x: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    gate: torch.Tensor,
    *,
    num_tiles: int,
) -> torch.Tensor:
    """Reference implementation matching trix.kernel.trix_forward semantics."""
    batch, in_features = x.shape
    out_features = weights.shape[0]
    out = (x.to(torch.float32) @ weights.t().to(torch.float32)) * scales.to(
        torch.float32
    )

    out_per_tile = out_features // num_tiles
    for t in range(num_tiles):
        s = t * out_per_tile
        e = s + out_per_tile
        out[:, s:e] = out[:, s:e] * gate[:, t : t + 1].to(torch.float32)
    return out


def test_pack_weights_matches_python_fallback_when_native_present_or_not():
    from trix.kernel import pack_weights, unpack_weights
    from trix.kernel import bindings as b

    w = _rand_ternary(17, 259, seed=0)
    packed = pack_weights(w)
    packed_py = b._pack_weights_python(w)
    assert torch.equal(packed.cpu(), packed_py.cpu())

    w2 = unpack_weights(packed, 17, 259)
    assert torch.equal(w.cpu(), w2.cpu())


def test_trix_forward_matches_reference():
    from trix.kernel import pack_weights, trix_forward

    torch.manual_seed(0)
    batch, in_f, out_f, num_tiles = 8, 33, 64, 4
    assert out_f % num_tiles == 0

    x = torch.randn(batch, in_f)
    w = _rand_ternary(out_f, in_f, seed=1)
    scales = torch.randn(out_f).abs() + 0.1
    gate = torch.randint(0, 2, (batch, num_tiles), dtype=torch.int64).to(torch.float32)

    packed = pack_weights(w)
    out = trix_forward(x, packed, scales, gate, out_f, num_tiles)
    ref = _trix_forward_reference(x, w, scales, gate, num_tiles=num_tiles)
    assert torch.allclose(out.cpu(), ref.cpu(), atol=1e-5)


@pytest.mark.parametrize(
    "batch,in_f,out_f,num_tiles",
    [
        (1, 7, 16, 4),
        (4, 33, 64, 4),
        (8, 128, 128, 4),
        (16, 127, 256, 8),
    ],
)
def test_trix_forward_matches_reference_various_shapes(batch, in_f, out_f, num_tiles):
    from trix.kernel import pack_weights, trix_forward

    torch.manual_seed(0)
    assert out_f % num_tiles == 0

    x = torch.randn(batch, in_f)
    w = _rand_ternary(out_f, in_f, seed=123)
    scales = torch.randn(out_f).abs() + 0.1
    gate = torch.randint(0, 2, (batch, num_tiles), dtype=torch.int64).to(torch.float32)

    packed = pack_weights(w)
    out = trix_forward(x, packed, scales, gate, out_f, num_tiles)
    ref = _trix_forward_reference(x, w, scales, gate, num_tiles=num_tiles)
    assert torch.allclose(out.cpu(), ref.cpu(), atol=1e-5)


@pytest.mark.skipif(
    not __import__("trix.kernel").is_neon_available(),
    reason="native library not available",
)
def test_native_pack_unpack_and_forward_match_reference_strict():
    """If native library exists, enforce native-vs-reference equivalence."""
    from trix.kernel import bindings as b
    from trix.kernel import pack_weights, unpack_weights, trix_forward

    # Force native load.
    lib = b._load_library(raise_on_missing=True)
    assert lib is not None

    batch, in_f, out_f, num_tiles = 4, 128, 128, 4
    w = _rand_ternary(out_f, in_f, seed=2)
    packed_native = pack_weights(w)
    packed_py = b._pack_weights_python(w)
    assert torch.equal(packed_native.cpu(), packed_py.cpu())

    w2 = unpack_weights(packed_native, out_f, in_f)
    assert torch.equal(w.cpu(), w2.cpu())

    x = torch.randn(batch, in_f)
    scales = torch.ones(out_f)
    gate = torch.tensor([[1, 0, 1, 0]] * batch, dtype=torch.float32)

    out = trix_forward(x, packed_native, scales, gate, out_f, num_tiles)
    ref = _trix_forward_reference(x, w, scales, gate, num_tiles=num_tiles)
    assert torch.allclose(out.cpu(), ref.cpu(), atol=1e-5)
