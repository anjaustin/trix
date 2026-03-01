# TriX Benchmarks

Reproducible performance benchmarks for the TriX architecture.

## Table of Contents

1. [Methodology](#methodology)
2. [TinyShakespeare Results](#tinyshakespeare-results)
3. [Memory Benchmarks](#memory-benchmarks)
4. [Routing Analysis](#routing-analysis)
5. [Reproducing Results](#reproducing-results)

---

## Methodology

### Principles

All benchmarks follow these principles:

1. **Reproducibility**: Fixed seeds, documented configurations
2. **Fair Comparison**: Same hyperparameters across architectures where applicable
3. **Statistical Validity**: Multiple runs, reported variance
4. **Transparency**: Full code provided

### Hardware

Primary benchmarks run on:
- **NVIDIA Jetson AGX Orin** (64GB)
- **CPU Baseline**: Intel Xeon (for reference)

Results should be reproducible on any CUDA-capable GPU.

### Software

- Python 3.10+
- PyTorch 2.0+
- NumPy 1.24+

---

## TinyShakespeare Results

### Task Description

Character-level language modeling on the TinyShakespeare dataset (~1MB of Shakespeare text).

**Metric**: Validation perplexity (lower is better)

### Configuration

```python
config = {
    'd_model': 128,
    'n_layers': 4,
    'n_heads': 4,
    'dropout': 0.1,
    'batch_size': 64,
    'seq_len': 256,
    'epochs': 10,
    'lr': 3e-4,
    'seed': 42,
}
```

### Results

| Model | Tiles | Parameters | Val PPL | Δ Baseline |
|-------|-------|------------|---------|------------|
| Dense FFN (baseline) | — | 1,052,672 | 20.14 | — |
| SparseTriXFFN | 4 | 892,416 | 19.26 | −4.4% |
| HierarchicalTriXFFN | 16 | 826,304 | 17.16 | −14.8% |
| HierarchicalTriXFFN | 64 | 612,352 | 16.89 | −16.1% |
| **SparseLookupFFN** | **64** | **366,412** | **16.56** | **−17.8%** |

### Key Findings

1. **More tiles improve quality** up to ~64 tiles
2. **SparseLookup achieves best results** with fewest parameters
3. **2.9× parameter reduction** vs baseline with better perplexity

### Statistical Significance

Results over 5 seeds (mean ± std):

| Model | Val PPL |
|-------|---------|
| Dense FFN | 20.14 ± 0.23 |
| SparseLookupFFN-64 | 16.56 ± 0.18 |

p-value < 0.001 (paired t-test)

---

## Memory Benchmarks

### Weight Memory

| Representation | Model Size | Compression |
|----------------|------------|-------------|
| FP32 | 4.00 MB | 1.0× |
| FP16 | 2.00 MB | 2.0× |
| INT8 | 1.00 MB | 4.0× |
| **TriX 2-bit** | **0.25 MB** | **16.0×** |

*Measured for HierarchicalTriXFFN with d_model=512, num_tiles=64*

### Activation Memory

Sparse computation reduces activation memory:

| Model | Peak Activation Memory |
|-------|------------------------|
| Dense FFN | 100% (baseline) |
| TriX (64 tiles) | ~1.5% |

*Only winning tile activations stored*

### Inference Memory

```
Total Inference Memory = Model Weights + Activations + Workspace

Dense:  4.00 MB + 8.00 MB + 0.5 MB = 12.5 MB
TriX:   0.25 MB + 0.12 MB + 0.5 MB =  0.9 MB
```

---

## Routing Analysis

### Tile Utilization

Healthy routing shows uniform tile usage:

```
Ideal:    [6.25%, 6.25%, 6.25%, ...] for 16 tiles
Observed: [5.8%, 6.1%, 6.4%, 5.9%, ...] (std: 0.4%)
```

### Signature Diversity

Pairwise Hamming distance between signatures:

| Configuration | Mean Distance | Min | Max |
|---------------|---------------|-----|-----|
| Random init | 50.2% | 41% | 58% |
| After training | 48.7% | 32% | 67% |

Training maintains diversity while allowing some specialization.

### Routing Stability

Fraction of inputs that route to the same tile across epochs:

| Epoch | Stability |
|-------|-----------|
| 1-2 | 45% |
| 5-6 | 78% |
| 9-10 | 94% |

Routing stabilizes as training progresses.

---

## Reproducing Results

### Quick Benchmark

```bash
cd TriXO
python scripts/benchmark_ffn.py
```

Expected output:
```
================================================================================
                         TriX FFN Benchmark Suite
================================================================================

Dataset: TinyShakespeare (1,115,394 chars)
Config:  d_model=128, n_layers=4, epochs=10, seed=42

--------------------------------------------------------------------------------
Training Dense Baseline...
  Epoch 10/10: train_loss=1.842, val_ppl=20.14
  Time: 45.2s

Training SparseTriXFFN (4 tiles)...
  Epoch 10/10: train_loss=1.793, val_ppl=19.26
  Time: 38.1s

Training HierarchicalTriXFFN (64 tiles)...
  Epoch 10/10: train_loss=1.721, val_ppl=16.89
  Time: 52.3s

Training SparseLookupFFN (64 tiles)...
  Epoch 10/10: train_loss=1.698, val_ppl=16.56
  Time: 41.7s

================================================================================
                              RESULTS SUMMARY
================================================================================

| Model                 | Params    | Val PPL | vs Baseline |
|-----------------------|-----------|---------|-------------|
| Dense FFN             | 1,052,672 | 20.14   | —           |
| SparseTriXFFN-4       | 892,416   | 19.26   | −4.4%       |
| HierarchicalTriXFFN-64| 612,352   | 16.89   | −16.1%      |
| SparseLookupFFN-64    | 366,412   | 16.56   | −17.8%      |

================================================================================
```

### Custom Benchmark

```python
from scripts.benchmark_ffn import run_benchmark

results = run_benchmark(
    d_model=256,
    num_tiles=128,
    n_layers=6,
    epochs=20,
    seeds=[42, 123, 456],  # Multiple seeds
)

for model_name, metrics in results.items():
    print(f"{model_name}: {metrics['val_ppl']:.2f} ± {metrics['val_ppl_std']:.2f}")
```

### Memory Benchmark

```python
import torch
from trix import HierarchicalTriXFFN, pack_weights

ffn = HierarchicalTriXFFN(d_model=512, num_tiles=64)

# FP32 size
fp32_size = sum(p.numel() * 4 for p in ffn.parameters())

# Packed size
packed = ffn.pack_weights()
packed_size = sum(p.numel() for p in packed.values())

print(f"FP32:   {fp32_size / 1024 / 1024:.2f} MB")
print(f"Packed: {packed_size / 1024 / 1024:.2f} MB")
print(f"Ratio:  {fp32_size / packed_size:.1f}×")
```

---

## Benchmark Scripts

### Available Scripts

| Script | Purpose |
|--------|---------|
| `scripts/benchmark_ffn.py` | Main FFN comparison |
| `scripts/run_validation.py` | Validate claims |

### Adding New Benchmarks

```python
# scripts/benchmark_custom.py
import torch
from trix import HierarchicalTriXFFN

def benchmark_latency(model, input_shape, warmup=10, runs=100):
    """Benchmark inference latency."""
    x = torch.randn(*input_shape)
    
    # Warmup
    for _ in range(warmup):
        _ = model(x)
    
    # Timed runs
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(runs):
        _ = model(x)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / runs
    
    return elapsed_ms
```

---

## Known Limitations

1. **Dataset Scale**: Current benchmarks use small datasets (TinyShakespeare). Large-scale validation pending.

2. **Hardware Variance**: Results may vary by ±5% across hardware platforms.

3. **Training Variance**: Different random seeds can affect final perplexity by ±0.3.

4. **NEON Acceleration**: ARM-specific optimizations not reflected in CUDA benchmarks.

---

## Reporting Issues

If you cannot reproduce results:

1. Check Python/PyTorch versions match
2. Verify random seed is set correctly
3. Ensure dataset downloads correctly
4. Open an issue with full configuration and output
