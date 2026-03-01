import torch


def test_falsify_flat_popcount_not_equivalent_to_dot_on_continuous_inputs():
    """Counterexample: same sign pattern, different magnitudes.

    flat_popcount uses sign(x) (magnitude-blind). Dot routing uses magnitudes.
    Two vectors with identical sign patterns can route differently under dot,
    but identically under flat_popcount.
    """

    from trix.nn import SparseLookupFFNv2

    ffn = SparseLookupFFNv2(
        d_model=4,
        num_tiles=4,
        tiles_per_cluster=4,  # disable clustering (single cluster)
        dropout=0.0,
        use_score_calibration=False,
    )
    ffn.eval()

    sigs = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    with torch.no_grad():
        ffn.signatures_raw.copy_(sigs)

    x1 = torch.tensor([[10.0, 10.0, 1.0, 1.0]])
    x2 = torch.tensor([[10.0, 1.0, 10.0, 1.0]])

    # Dot routing diverges.
    idx1_dot, _ = ffn.route(
        x1, ffn.signatures, backend="hierarchical_dot", return_scores=False
    )
    idx2_dot, _ = ffn.route(
        x2, ffn.signatures, backend="hierarchical_dot", return_scores=False
    )
    assert int(idx1_dot.item()) == 0
    assert int(idx2_dot.item()) == 1

    # Popcount routing ties and chooses the first tile consistently.
    idx1_pc, _ = ffn.route(
        x1, ffn.signatures, backend="flat_popcount", return_scores=False
    )
    idx2_pc, _ = ffn.route(
        x2, ffn.signatures, backend="flat_popcount", return_scores=False
    )
    assert int(idx1_pc.item()) == 0
    assert int(idx2_pc.item()) == 0


def test_falsify_contracted_mode_requires_class_hint():
    """Counterexample: contracted mode doesn't help if you don't supply class_hint."""

    from trix.nn import DropInFFN, DropInConfig

    torch.manual_seed(0)
    m = DropInFFN(
        DropInConfig(d_model=16, num_tiles=4, tiles_per_cluster=2), mode="contracted"
    )

    # Populate claims so compile_stable can compile something.
    x = torch.randn(4, 8, 16)
    labels = torch.randint(0, 4, (4, 8))
    _out, _info, _aux = m.ffn(x, labels=labels)
    m.compile_stable(threshold=0.0, min_confidence=0.0, num_classes=4)

    # Without class_hint, forward falls back to dynamic.
    out, info, _aux = m(x, return_aux=True)
    assert info.get("compiled") is False
    assert out.shape == x.shape
