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

Note:
- `trix bench` runs scripts from `experiments/` and therefore expects a repo checkout (editable install).

On success it prints a short summary and writes:
- `suite_v1.json`
- `routing_telemetry.jsonl`
- `drift_telemetry.jsonl`

## Bundles

Bundles are portable routing artifacts (compressed signatures + optional compiled dispatch + validation report).

Note: bundle export/import currently supports `SparseLookupFFNv2` bundles.

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

When `--validate` is set, `load-bundle` also:
- prints a Mesa 15 compatibility report
- verifies `manifest.json` if present

Integrity:

```bash
trix bundle manifest --outdir bundle_dir
trix bundle verify --outdir bundle_dir
```

Drift policy check (against a suite output):

```bash
trix drift check --suite results/benchmarks_v1/suite_v1.json --policy drift_policy.json
```
