import torch


def test_pack_weights_with_alpha_roundtrip_and_forward():
    from trix.kernel import pack_weights_with_alpha, unpack_weights, trix_forward

    torch.manual_seed(0)

    batch, in_f, out_f, num_tiles = 4, 33, 64, 4
    assert out_f % num_tiles == 0

    x = torch.randn(batch, in_f)
    w = torch.randn(out_f, in_f)

    packed, scales, ternary = pack_weights_with_alpha(w, threshold=0.5)

    # pack/unpack matches ternary
    u = unpack_weights(packed, out_f, in_f)
    assert torch.equal(u.cpu(), ternary.cpu())

    gate = torch.randint(0, 2, (batch, num_tiles), dtype=torch.int64).to(torch.float32)
    out = trix_forward(x, packed, scales, gate, out_f, num_tiles)

    # Reference
    ref = (x.to(torch.float32) @ ternary.t().to(torch.float32)) * scales.to(
        torch.float32
    )
    out_per_tile = out_f // num_tiles
    for t in range(num_tiles):
        ref[:, t * out_per_tile : (t + 1) * out_per_tile] *= gate[:, t : t + 1]

    assert torch.allclose(out.cpu(), ref.cpu(), atol=1e-5)
