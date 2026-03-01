import torch


def test_falsify_alpha_mean_abs_sparse_outlier_is_bad_approximation():
    """Counterexample: mean(abs) alpha can be pathological for sparse outliers.

    If a row has a single large weight and many zeros, alpha=mean(abs)
    is diluted by the zeros. The ternary+alpha approximation can be off
    by orders of magnitude.
    """

    from trix.kernel import pack_weights_with_alpha, trix_forward

    batch, in_f, out_f, num_tiles = 1, 128, 128, 4
    assert out_f % num_tiles == 0

    w = torch.zeros(out_f, in_f)
    w[0, 0] = 100.0

    packed, scales, ternary = pack_weights_with_alpha(w, threshold=0.5)

    x = torch.zeros(batch, in_f)
    x[0, 0] = 1.0
    gate = torch.ones(batch, num_tiles)

    y = trix_forward(x, packed, scales, gate, out_f, num_tiles)

    # True dense value (no approximation): 100
    expected = 100.0
    approx = float(y[0, 0].item())

    # Relative error should be huge (> 90%).
    rel_err = abs(approx - expected) / expected
    assert rel_err > 0.9

    # Sanity: ternary kept the sign at (0,0)
    assert float(ternary[0, 0].item()) == 1.0


def test_falsify_alpha_discontinuity_near_threshold():
    """Counterexample: tiny weight changes around threshold can flip ternary codes.

    This shows alpha+ternary quantization is discontinuous: two weight matrices
    that are extremely close can yield different packed weights and therefore
    different outputs.
    """

    from trix.kernel import pack_weights_with_alpha, trix_forward

    batch, in_f, out_f, num_tiles = 1, 16, 16, 4
    assert out_f % num_tiles == 0

    # Two nearly identical weights around threshold
    w0 = torch.zeros(out_f, in_f)
    w1 = torch.zeros(out_f, in_f)
    w0[0, 0] = 0.49
    w1[0, 0] = 0.51

    p0, s0, _t0 = pack_weights_with_alpha(w0, threshold=0.5)
    p1, s1, _t1 = pack_weights_with_alpha(w1, threshold=0.5)

    x = torch.zeros(batch, in_f)
    x[0, 0] = 1.0
    gate = torch.ones(batch, num_tiles)

    y0 = trix_forward(x, p0, s0, gate, out_f, num_tiles)
    y1 = trix_forward(x, p1, s1, gate, out_f, num_tiles)

    # Weight changed by 0.02, but output changes discretely.
    assert float(y0[0, 0].item()) != float(y1[0, 0].item())
