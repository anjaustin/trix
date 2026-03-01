import torch


def test_lifecycle_margin_can_miss_cluster_ties():
    """Falsification: lifecycle margin/tie stats do not universally reflect cluster-level ties.

    SparseLookupFFNv2 routing is hierarchical:
    - choose a cluster using cluster signatures
    - then choose a tile within that cluster

    The lifecycle v1 observe() currently computes tie/margin over the score
    columns that were populated during routing (cluster-local). This can miss
    cluster-level tie degeneracy.
    """

    from trix.nn import SparseLookupFFNv2, RoutingLifecycleV1

    torch.manual_seed(0)
    ffn = SparseLookupFFNv2(
        d_model=4,
        num_tiles=4,
        tiles_per_cluster=2,
        dropout=0.0,
        use_score_calibration=False,
    )
    ffn.eval()

    # Construct signatures such that BOTH clusters have cluster signature == 0.
    # Cluster 0 tiles: [1,0,0,0] and [-1,0,0,0] -> mean [0,0,0,0] -> sign [0,0,0,0]
    # Cluster 1 tiles: [1,0,0,0] and [-1,0,0,0] -> same
    sig = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ]
    )
    with torch.no_grad():
        ffn.signatures_raw.copy_(sig)

    lc = RoutingLifecycleV1(ffn)
    x = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # B=1,T=1,D=4

    _out, _info, _aux, rec = lc.observe(x)

    # Cluster scores are identical (both 0), i.e. cluster tie degeneracy exists.
    # But within the selected cluster, tile scores are [1, -1] -> margin 2.
    assert rec["tie_rate"] == 0.0
    assert rec["margin_mean"] > 1.0


def test_score_calibration_can_break_score_ordering():
    """Falsification: score calibration spline is learnable and can become non-monotonic.

    If non-monotonic, argmax(calibrator(scores)) can differ from argmax(scores).
    """

    from trix.nn import SparseLookupFFNv2

    torch.manual_seed(0)
    ffn = SparseLookupFFNv2(
        d_model=8,
        num_tiles=4,
        tiles_per_cluster=2,
        dropout=0.0,
        use_score_calibration=True,
    )
    ffn.eval()

    # Force calibrator knot values to be decreasing (strongly non-monotonic).
    with torch.no_grad():
        ffn.score_calibrator.knot_values.copy_(
            torch.linspace(1.0, 0.0, ffn.score_calibrator.num_knots)
        )
        ffn.score_calibrator.temperature.fill_(1.0)

    # Two scores s0 < s1
    scores = torch.tensor([[0.1, 0.2]])
    gates = ffn.score_calibrator(scores)

    # If mapping is non-monotonic enough, ordering can flip.
    # We don't require it flips for all values, but we assert it's possible by construction.
    assert (scores.argmax(dim=-1) != gates.argmax(dim=-1)).item() is True
