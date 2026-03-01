# CLI

TriX includes a small CLI intended to make the first-run experience simple.

## Doctor

Environment + correctness self-check:

```bash
trix doctor
```

## Bench

Run the canonical benchmark suite v1 (writes JSON + JSONL artifacts):

```bash
trix bench --outdir results/benchmarks_v1 --device cpu
```

## Bundles

Bundles are portable routing artifacts (compressed signatures + optional compiled dispatch + validation report).

Export from a `SparseLookupFFNv2` state dict:

```bash
trix export-bundle --config bundle_config.json --state-dict state_dict.pt --outdir bundle_dir --include-state --compile-stable --validate
```

Example `bundle_config.json`:

```json
{
  "ffn_type": "SparseLookupFFNv2",
  "d_model": 64,
  "num_tiles": 8,
  "tiles_per_cluster": 4,
  "grid_size": 16,
  "use_score_calibration": false,
  "routing_backend": "hierarchical_dot",
  "ternary_weight": 0.05,
  "sparsity_weight": 0.05,
  "diversity_weight": 0.05
}
```

Load and validate:

```bash
trix load-bundle --outdir bundle_dir --validate
```
