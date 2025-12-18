# Riemann Zero Hunter Experiments

> **⚠️ EXPERIMENTAL** - These experiments explore Riemann zero computation. Results vary significantly based on parameters.

## Overview

This index covers all Riemann-related experiments across the repository.

## Main Implementation

Located in `experiments/number_theory/trix_riemann/`

### Core Tiles

| File | Purpose | Status |
|------|---------|--------|
| `theta_tile.py` | θ(t) computation | ✓ Complete |
| `dirichlet_tile.py` | Coefficient generation | ✓ Complete |
| `spectral_tile.py` | FFT + TriXFFT | ✓ Complete |
| `sign_tile.py` | Zero detection | ✓ Complete |
| `probe.py` | 100% TriX pipeline | ✓ Complete |

### Engines

| File | Purpose | Best Rate |
|------|---------|-----------|
| `adaptive_engine.py` | Auto-switching hybrid/FP64 | 13.6M/s @ t=10^10 |
| `fused_riemann_engine.py` | Fused evaluation | 24M/s (coarse) |
| `odlyzko_engine.py` | O-S algorithm | 121K/s (overhead) |
| `riemann_hunter.py` | Autonomous hunt | 270K/s |

### FFT Implementations

| File | Purpose | Notes |
|------|---------|-------|
| `triton_fft.py` | N=8 Triton kernel | 15x speedup |
| `triton_fft_large.py` | Stockham for large N | Working |
| `triton_fft_thor.py` | Thor optimization | 813K FFTs/s |
| `hollywood_fft.py` | Topology-as-algorithm | Research |
| `hollywood_triton_fused.py` | Fused architecture | Research |

### Advanced Algorithms

| File | Purpose | Notes |
|------|---------|-------|
| `chirp_tile.py` | Chirp-Z Transform | In progress |
| `chirp_fft.py` | Chirp FFT | In progress |
| `fused_nufft.py` | NUFFT spreading | 100% detection |
| `hollywood_gather_nufft.py` | Gather vs scatter | 4.77x speedup |
| `range_fft.py` | Range evaluation | Research |

## Tesseract Experiments

Located in `experiments/`

| File | Purpose | Result |
|------|---------|--------|
| `riemann_tesseract.py` | 29D hypercube framework | 220 zeros/s |
| `riemann_tesseract_parallel.py` | True parallel eval | 21K zeros/s |
| `riemann_tesseract_prove.py` | Reality benchmark | 0.45M/s |
| `riemann_tesseract_xor.py` | XOR evaluation | Research |
| `riemann_parallel_final.py` | Final benchmark | 0.45M/s |
| `riemann_fft.py` | torch.fft approach | 11M/s small t |
| `riemann_chunked.py` | Memory efficient | 6.8M/s |
| `riemann_triton_kernel.py` | Custom Triton | 0.52M/s |
| `riemann_triton_v2.py` | Matmul + FP16 | OOM |
| `riemann_torch_fft.py` | Optimized torch | 0.3M/s |

## Performance Summary

### By Height t

| Height | M terms | Best Rate | Method |
|--------|---------|-----------|--------|
| 10^8 | 4,000 | 11M/s | Direct matmul |
| 10^10 | 40,000 | 13.6M/s | Adaptive hybrid |
| 10^12 | 400,000 | 1.8M/s | Adaptive FP64 |
| 10^13 | 1.26M | 1.7M/s | Adaptive FP64 |

### Projections

| Target | Time (Current) | Time (with O-S) |
|--------|---------------|-----------------|
| 10^10 | 2.1 hours | Minutes |
| 10^12 | 6.4 days | Hours |
| 10^13 | 68 days | ~10 minutes |
| 10^14 | 2 years | ~1 hour |

## Key Findings

### What Works

1. **Hybrid Precision**: FP64 phases + FP32 bulk = best throughput
2. **NUFFT Spreading**: B-spline achieves 100% zero detection
3. **Adaptive Switching**: Auto mode selection based on t
4. **Sign Detection**: 3.76B/s for INT8 comparison

### What Doesn't

1. **XOR for Z(t)**: Transcendentals don't map to INT8
2. **Pure Triton**: Overhead exceeds torch.fft
3. **Large Batches**: Memory bandwidth saturates
4. **Direct Method at Scale**: O(N×M) kills performance

### The Fundamental Limit

```
M = √(t / 2π)
```

At t=10^13, each Z(t) evaluation requires 1.26 million terms.
No clever coding avoids this. Only Odlyzko-Schönhage reduces complexity.

## Documentation

| File | Purpose |
|------|---------|
| `README.md` | Architecture overview |
| `DISCOVERY_LOG.md` | Research journal |
| `SPEC_ODLYZKO_SCHONHAGE.md` | O-S algorithm spec |
| `docs/RIEMANN_REALITY.md` | Honest assessment |
| `docs/MESA_10_RIEMANN_PROBE.md` | Mesa 10 overview |

## Running Experiments

```bash
# Adaptive engine benchmark
python experiments/number_theory/trix_riemann/adaptive_engine.py

# Reality check
python experiments/riemann_parallel_final.py

# Tesseract framework
python experiments/riemann_tesseract.py
```

## Conclusion

The Riemann Zero Hunter achieved:
- Peak: 13.6M zeros/sec at t=10^10
- Sustained: 1.7M zeros/sec at t=10^13
- Projection: 68 days for 10^13 zeros

The tesseract architecture excels at discrete operations (6502 = 100%) but transcendentals limit Riemann performance.

**Next step**: Complete Odlyzko-Schönhage implementation for O(M log M) complexity.
