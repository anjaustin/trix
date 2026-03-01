import torch


def _rand_pm1(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(0, 2, (n, d), generator=g)
    x = torch.empty((n, d), dtype=torch.float32)
    x[r == 0] = -1.0
    x[r == 1] = 1.0
    return x


def test_flat_popcount_routing_matches_dot_on_pm1_inputs():
    """Property: on x in {+1,-1}^d, dot-argmax matches popcount-argmin."""

    from trix.nn import SparseLookupFFNv2

    torch.manual_seed(0)
    d_model = 257
    num_tiles = 16
    ffn = SparseLookupFFNv2(
        d_model=d_model,
        num_tiles=num_tiles,
        tiles_per_cluster=4,
        dropout=0.0,
        use_score_calibration=False,
    )

    x = _rand_pm1(128, d_model, seed=1)
    sigs = torch.sign(torch.randn(num_tiles, d_model))

    # Dot routing over ternary signatures with pm1 input
    scores = x @ sigs.t()
    win_dot = scores.argmax(dim=-1)

    win_pop, _scores_pop = ffn.route(
        x, sigs, return_scores=True, backend="flat_popcount"
    )
    assert torch.equal(win_dot, win_pop)


def test_flat_popcount_returns_dense_scores():
    from trix.nn import SparseLookupFFNv2

    d_model = 33
    num_tiles = 8
    ffn = SparseLookupFFNv2(
        d_model=d_model,
        num_tiles=num_tiles,
        tiles_per_cluster=4,
        dropout=0.0,
        use_score_calibration=False,
    )
    x = _rand_pm1(10, d_model, seed=0)
    sigs = torch.sign(torch.randn(num_tiles, d_model))

    _idx, scores = ffn.route(x, sigs, return_scores=True, backend="flat_popcount")
    assert scores is not None
    assert scores.shape == (10, num_tiles)
