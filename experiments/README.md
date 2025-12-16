# TriX Training Experiments

## Overview

**Date:** 2025-12-15  
**Hardware:** NVIDIA Jetson AGX Thor (Blackwell)  
**GPU Memory:** 128 GB Unified  
**Compute:** 2070 FP4 TFLOPS  

## Experiment: Parallel Multi-Scale Training

### Objective

Validate TriX SparseLookupFFN across multiple scales and domains:
1. Demonstrate quality improvements over standard FFN
2. Characterize training dynamics
3. Establish baseline metrics for future work

### Models

| ID | Name | Dataset | Tokens | Params | Architecture |
|----|------|---------|--------|--------|--------------|
| E1 | trix-tiny | FineWeb-Edu | 5M | ~2.5M | SparseLookupFFN |
| E2 | trix-small | FineWeb-Edu | 50M | ~10M | SparseLookupFFN |
| E3 | trix-medium | FineWeb-Edu | 500M | ~40M | SparseLookupFFN |
| E4 | trix-code | The Stack v2 | 250M | ~40M | SparseLookupFFN |

### Hypothesis

TriX SparseLookupFFN will achieve lower perplexity than parameter-equivalent standard FFN models due to:
1. Emergent routing enabling input-dependent specialization
2. Sparse computation focusing capacity where needed
3. Ternary weight structure acting as implicit regularization

### Metrics Tracked

**Per Step:**
- Training loss
- Learning rate
- Gradient norm
- Throughput (tokens/sec)
- Memory usage
- Routing entropy (tile utilization)
- Auxiliary losses (balance, diversity)

**Per Epoch:**
- Validation loss
- Validation perplexity
- Tile usage histogram
- Routing stability (% tokens changing routes)
- Checkpoint saved

**Final:**
- Best validation perplexity
- Total training time
- Tokens processed
- Final routing statistics

### Hardware Configuration

```yaml
device: NVIDIA Thor
memory: 128 GB unified
cuda_version: "13.0"
pytorch_version: "2.8.0"
compute_capability: "11.0"
precision: BF16 (AMP)
```

### Reproducibility

```yaml
seed: 42
deterministic: true
benchmark: false
```

---

## File Structure

```
experiments/
├── README.md                 # This file
├── configs/                  # Model configurations
│   ├── tiny.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   └── code.yaml
├── data/                     # Prepared datasets
│   ├── fineweb_5m/
│   ├── fineweb_50m/
│   ├── fineweb_500m/
│   └── stack_250m/
├── logs/                     # Training logs
│   ├── tiny/
│   ├── small/
│   ├── medium/
│   └── code/
├── checkpoints/              # Model checkpoints
│   ├── tiny/
│   ├── small/
│   ├── medium/
│   └── code/
└── results/                  # Final results and analysis
    ├── metrics.json
    ├── plots/
    └── REPORT.md
```

---

## Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Data Prep | ~30 min | Download and tokenize datasets |
| Training | ~6 hrs | All 4 models in parallel |
| Analysis | ~30 min | Generate reports and plots |
| **Total** | **~7 hrs** | |

---

## Contact

Repository: https://github.com/trix/trix  
Hardware Partner: NVIDIA (Jetson AGX Thor)
