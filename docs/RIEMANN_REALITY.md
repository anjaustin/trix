# Riemann Reality: Honest Performance Assessment

> **Documenting what we learned about Riemann zero computation.**

## The Goal

Compute 10^13 Riemann zeta zeros to match/exceed the world record.

## The Claims vs Reality

### Original Claims

| Target | Claimed Time | Basis |
|--------|--------------|-------|
| 10^13 zeros | 6 seconds | 275 TOPS INT8 |
| 10^14 zeros | 1 minute | XOR routing |

### Measured Reality

| Target | Actual Time | Why |
|--------|-------------|-----|
| 10^13 zeros | **68-258 days** | FP32 transcendentals |
| 10^14 zeros | **2-7 years** | sqrt(t) scaling |

**The gap: 1,000,000x**

## Why the Gap?

### The Riemann-Siegel Formula

```
Z(t) = 2 × Σ_{n=1}^{M} cos(t·ln(n) - θ(t)) / √n
```

Where M = √(t / 2π).

### The Problem

1. **Transcendentals**: Z(t) requires `cos()`, `log()`, `sqrt()`
2. **M scaling**: At t=10^13, M = 1,261,576 terms per evaluation
3. **No XOR**: These operations don't map to INT8

### What 275 TOPS Actually Means

| Operation | TOPS | Reality |
|-----------|------|---------|
| INT8 XOR/ADD | 275 | ✓ Achievable |
| FP32 MUL | ~67 | GPU limit |
| FP32 cos() | ~10 | Special function unit |
| **Riemann Z(t)** | **~0.001** | M multiplies + transcendentals |

## Measured Rates

### On NVIDIA Thor (131.9 GB, 275 TOPS INT8)

| Implementation | Rate | 10^13 Time |
|----------------|------|------------|
| Pure Python | 5.5K/sec | 57 years |
| PyTorch matmul | 11M/sec (small t) | — |
| Adaptive Engine (t=10^10) | 13.6M/sec | — |
| Adaptive Engine (t=10^13) | **1.7M/sec** | **68 days** |
| Extended run (sustained) | **0.45M/sec** | **258 days** |

### The sqrt(t) Scaling

| Height t | M terms | Measured Rate |
|----------|---------|---------------|
| 10^8 | 4,000 | 11M/sec |
| 10^10 | 40,000 | 13M/sec (peak) |
| 10^12 | 400,000 | 1.8M/sec |
| 10^13 | 1,261,576 | 1.7M/sec |

Rate scales as ~1/√t after t > 10^10.

## What Would Actually Work

### Odlyzko-Schönhage Algorithm

Uses Chirp-Z Transform to evaluate M points in O(M log M) instead of O(N × M).

| Method | Complexity | Speedup at t=10^16 |
|--------|------------|-------------------|
| Direct | O(N × M) | 1x |
| Chirp/O-S | O(M log M) | ~4,000,000x |

**But**: Still requires FFT (cos/sin), just fewer of them.

### Realistic Projections with O-S

| Target | Time (10 TFLOPS) | Time (100 TFLOPS) |
|--------|------------------|-------------------|
| 10^13 | 11 minutes | 1 minute |
| 10^14 | 1.5 hours | 9 minutes |
| 10^16 | 9 days | 22 hours |

**This is achievable** with proper implementation.

## The Tesseract's Actual Strengths

### Where XOR Excels (3.76 billion ops/sec)

| Operation | Speed | Example |
|-----------|-------|---------|
| Sign detection | 3.76B/sec | Zero crossing |
| XOR routing | 3.76B/sec | Tile selection |
| Hamming distance | 3.76B/sec | Nearest neighbor |
| Integer comparison | 3.76B/sec | Threshold |

### Where XOR Fails

| Operation | Why |
|-----------|-----|
| cos(x) | Transcendental, not XOR |
| log(x) | Transcendental, not XOR |
| sqrt(x) | Transcendental, not XOR |
| Floating point | Different instruction set |

## Honest Conclusions

### What We Learned

1. **INT8 TOPS ≠ FP32 TFLOPS**: Different operations, different limits
2. **sqrt(t) scaling is fundamental**: Can't XOR around Riemann-Siegel
3. **Tesseract excels at discrete ops**: 6502 = 100%, Riemann = slow
4. **Algorithm matters more than hardware**: O-S vs Direct = 4M× speedup

### Realistic Targets

| Goal | Approach | Time |
|------|----------|------|
| Beat world record (10^13) | Odlyzko-Schönhage | 1-10 minutes |
| Push to 10^14 | O-S + multi-GPU | 10-60 minutes |
| Reach 10^16 | O-S + cluster | 1-7 days |

### What We'd Need

1. **Complete O-S Implementation**: Chirp-Z working at scale
2. **Multi-GPU Parallelism**: Partition t-ranges across GPUs
3. **FP64 for Large t**: Precision matters at t > 10^14
4. **Verification Pipeline**: Riemann-von Mangoldt counts

## The Lesson

> "Match the tool to the problem."

- **Tesseract + XOR**: Perfect for 6502, state machines, binary nets
- **Riemann zeros**: Need Odlyzko-Schönhage, FP64, and patience

**The 6 seconds claim was wrong. 68 days is honest.**

## Files

| File | Purpose |
|------|---------|
| `experiments/riemann_parallel_final.py` | Honest benchmark |
| `experiments/riemann_tesseract_prove.py` | Reality check |
| `experiments/number_theory/trix_riemann/adaptive_engine.py` | Best rates |
| `experiments/number_theory/trix_riemann/odlyzko_engine.py` | O-S implementation |
