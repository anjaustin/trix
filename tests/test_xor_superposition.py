import torch


def _rand_ternary(*shape, seed: int):
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(0, 3, shape, generator=g)
    x = torch.zeros(shape, dtype=torch.float32)
    x[r == 0] = -1.0
    x[r == 2] = 1.0
    return x


def test_pack_unpack_roundtrip_dim_not_multiple_of_4():
    from trix.nn.xor_superposition import pack_ternary_to_uint8, unpack_uint8_to_ternary

    x = _rand_ternary(7, 257, seed=0)
    packed = pack_ternary_to_uint8(x)
    y = unpack_uint8_to_ternary(packed, dim=257)
    assert torch.equal(torch.sign(x), y)


def test_compress_decompress_lossless():
    from trix.nn.xor_superposition import CompressedSignatures

    sigs = _rand_ternary(32, 513, seed=1)
    comp = CompressedSignatures().compress(sigs)
    dec = comp.decompress_all()
    assert torch.equal(torch.sign(sigs), dec)


def test_dot_equals_masked_popcount_argmin():
    """Property: argmax dot equals argmin masked popcount distance."""
    from trix.nn.xor_superposition import (
        pack_ternary_to_uint8,
        popcount_distance_packed,
    )

    torch.manual_seed(0)
    dim = 257
    sigs = _rand_ternary(16, dim, seed=2)
    x = _rand_ternary(64, dim, seed=3)

    x_tern = torch.sign(x)
    s_tern = torch.sign(sigs)

    dot = x_tern @ s_tern.T
    win_dot = dot.argmax(dim=-1)

    px = pack_ternary_to_uint8(x_tern)
    ps = pack_ternary_to_uint8(s_tern)
    dist = popcount_distance_packed(px, ps, ignore_x_zeros=True)
    win_dist = dist.argmin(dim=-1)

    assert torch.equal(win_dot, win_dist)


def test_falsify_unmasked_popcount_with_zeros_can_disagree():
    """Counterexample: without masking, popcount distance can disagree when x has zeros."""
    from trix.nn.xor_superposition import (
        pack_ternary_to_uint8,
        popcount_distance_packed,
    )

    x = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    sigA = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    sigB = torch.tensor([[1.0, 1.0, -1.0, -1.0]])

    dotA = (x * sigA).sum(dim=-1)
    dotB = (x * sigB).sum(dim=-1)
    assert dotB.item() > dotA.item()

    px = pack_ternary_to_uint8(torch.sign(x))
    pA = pack_ternary_to_uint8(torch.sign(sigA))
    pB = pack_ternary_to_uint8(torch.sign(sigB))

    dA = popcount_distance_packed(px, pA, ignore_x_zeros=False)[0, 0].item()
    dB = popcount_distance_packed(px, pB, ignore_x_zeros=False)[0, 0].item()
    assert dA < dB
