# TriX Training Plan for Jetson AGX Thor

## Executive Summary

Train 3 language models on FineWeb-Edu subsets:
- **Tiny:** 5M tokens → ~250K params → ~2 minutes
- **Small:** 50M tokens → ~2.5M params → ~20 minutes  
- **Medium:** 500M tokens → ~25M params → ~3.5 hours

Total training time: **~4 hours**

---

## Hardware Profile (Measured)

| Spec | Value |
|------|-------|
| GPU | NVIDIA Thor |
| Memory | 122.8 GB unified |
| CUDA | 13.0 |
| Compute | SM 11.0 |
| Peak throughput | ~40K tok/s (AMP, batch=256) |

---

## Benchmark Results

### Standard FFN vs TriX FFN

| Model Size | Standard (tok/s) | TriX (tok/s) | Ratio |
|------------|------------------|--------------|-------|
| ~9M params | 105,174 | 76,293 | 0.73x |
| ~21M params | 59,907 | 37,639 | 0.63x |
| ~70M params | 29,846 | 12,803 | 0.43x |

### Optimization Impact

| Optimization | Throughput | Improvement |
|--------------|------------|-------------|
| FP32 baseline | 28,549 tok/s | 1.0x |
| AMP (FP16) | 39,879 tok/s | 1.4x |

### Recommendation

**Use Standard FFN with AMP for training.** TriX's advantage is inference-time memory compression (16x), not training speed. Train with standard FFN, then optionally convert to TriX for deployment.

---

## Model Configurations (Chinchilla-Optimal)

### Scaling Law

Chinchilla suggests `tokens ≈ 20 × parameters` for compute-optimal training.

### Config: Tiny (5M tokens)

```python
TINY_CONFIG = {
    'name': 'trix-tiny-5M',
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 3,
    'vocab_size': 8192,      # Smaller vocab for tiny dataset
    'seq_len': 256,
    'batch_size': 256,
    'learning_rate': 1e-3,
    'warmup_steps': 100,
    'total_tokens': 5_000_000,
    'params': '~250K',
}
# Estimated time: 5M / 100K tok/s = 50 seconds + overhead ≈ 2 minutes
```

### Config: Small (50M tokens)

```python
SMALL_CONFIG = {
    'name': 'trix-small-50M',
    'd_model': 256,
    'n_heads': 4,
    'n_layers': 6,
    'vocab_size': 16384,
    'seq_len': 256,
    'batch_size': 256,
    'learning_rate': 6e-4,
    'warmup_steps': 500,
    'total_tokens': 50_000_000,
    'params': '~2.5M',
}
# Estimated time: 50M / 60K tok/s = 830 seconds ≈ 15-20 minutes
```

### Config: Medium (500M tokens)

```python
MEDIUM_CONFIG = {
    'name': 'trix-medium-500M',
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 8,
    'vocab_size': 32000,
    'seq_len': 512,
    'batch_size': 128,        # Larger seq_len needs smaller batch
    'learning_rate': 3e-4,
    'warmup_steps': 2000,
    'total_tokens': 500_000_000,
    'params': '~25M',
}
# Estimated time: 500M / 40K tok/s = 12,500 seconds ≈ 3.5 hours
```

---

## Data Preparation

### Source: FineWeb-Edu

HuggingFace dataset with educational quality scores. We'll filter by `educational_score >= 3` and sample subsets.

### Tokenizer

Use GPT-2 tokenizer (BPE, 50K vocab) or train custom BPE for smaller vocabs on tiny/small.

### Subset Strategy

```
FineWeb-Edu (1.3T tokens)
    │
    ├── Filter: educational_score >= 3
    │
    ├── Sample: 500M tokens → medium_train.bin
    │
    ├── Sample: 50M tokens  → small_train.bin
    │
    └── Sample: 5M tokens   → tiny_train.bin
```

### Validation Split

5% held out for each: 250K / 2.5M / 25M tokens respectively.

---

## Training Schedule

### Phase 1: Tiny Model (5M tokens)
- **Purpose:** Validate training pipeline, catch bugs fast
- **Duration:** ~2 minutes
- **Checkpoints:** Every 1M tokens
- **Validation:** Every 500K tokens

### Phase 2: Small Model (50M tokens)  
- **Purpose:** Validate hyperparameters, learning rate schedule
- **Duration:** ~20 minutes
- **Checkpoints:** Every 10M tokens
- **Validation:** Every 5M tokens

### Phase 3: Medium Model (500M tokens)
- **Purpose:** Full training run, evaluate quality
- **Duration:** ~3.5 hours
- **Checkpoints:** Every 50M tokens
- **Validation:** Every 25M tokens

---

## Optimizations Applied

1. **AMP (Automatic Mixed Precision):** 1.4x speedup
2. **Large batch sizes:** 128-256 depending on seq_len
3. **Gradient accumulation:** If needed for effective batch size
4. **torch.compile:** For additional kernel fusion (if stable)
5. **Pin memory:** For faster CPU→GPU transfer
6. **Persistent workers:** For data loading

---

## Evaluation Metrics

1. **Perplexity (PPL):** Primary metric, on validation set
2. **Loss curve:** Should decrease smoothly
3. **Token throughput:** Should match benchmarks
4. **Memory usage:** Should stay under 80GB peak

---

## File Structure

```
/workspace/trix_latest/
├── data/
│   ├── tiny_train.bin      # 5M tokens
│   ├── tiny_val.bin        # 250K tokens
│   ├── small_train.bin     # 50M tokens
│   ├── small_val.bin       # 2.5M tokens
│   ├── medium_train.bin    # 500M tokens
│   └── medium_val.bin      # 25M tokens
├── checkpoints/
│   ├── tiny/
│   ├── small/
│   └── medium/
├── scripts/
│   ├── prepare_data.py     # Download & tokenize FineWeb-Edu
│   ├── train.py            # Main training script
│   └── benchmark_thor.py   # Hardware benchmarks (done)
└── configs/
    ├── tiny.yaml
    ├── small.yaml
    └── medium.yaml
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| OOM during training | Reduce batch size, use gradient checkpointing |
| Data download fails | Cache locally, use streaming fallback |
| Training diverges | Reduce LR, add gradient clipping |
| Slow throughput | Profile with torch.profiler, identify bottleneck |

---

## Next Steps

1. **Approve this plan**
2. **Implement data preparation script**
3. **Implement training script**
4. **Run Tiny model first** (validate everything works)
5. **Run Small model** (tune hyperparameters)
6. **Run Medium model** (full training)

---

## Questions for Review

1. **Vocab size:** Use standard GPT-2 (50K) or train smaller BPE?
2. **TriX integration:** Train standard FFN only, or also train TriX variants for comparison?
3. **Evaluation tasks:** Just perplexity, or add downstream tasks (HellaSwag, etc.)?
4. **Checkpointing frequency:** Current plan good, or more/less frequent?
