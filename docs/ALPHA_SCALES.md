# Alpha Scales (BitNet b1.58 Nugget)

TriX supports a BitNet-style pattern: ternary weights paired with per-row `alpha` scales.

API:
- `trix.kernel.pack_weights_with_alpha(...)`

## What Is Guaranteed

- Correctness relative to the declared ternary+alpha semantics:
  - `pack_weights_with_alpha` produces ternary weights and scales
  - `trix_forward(x, packed, scales, ...)` matches the reference computation using those ternary weights and scales

## What Is Not Guaranteed

- This is not a universal approximation guarantee for the original float weights.
- `alpha = mean(abs(w_row))` can be a poor fit depending on weight distribution.

## Falsification Cases (Expected Failure Modes)

We include explicit counterexamples:

1) Sparse outlier dilution
- If a row has one very large weight and many zeros, `mean(abs)` is diluted by zeros.
- The ternary+alpha approximation can be off by orders of magnitude.

2) Threshold discontinuity
- Tiny changes around the quantization threshold can flip a ternary code.
- This can produce discrete output changes.

Tests:
- `tests/test_kernel_alpha_falsify.py`

## Next Steps (If We Want Better Approximation)

- Consider alternative alpha definitions (e.g. mean over non-zeros, median(abs), clipped mean) and compare in benchmarks.
- Measure impact in `experiments/benchmarks/benchmark_suite_v1.py` (drift and stability).
