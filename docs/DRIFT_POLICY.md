# Drift Policy (Mesa 15)

Mesa 15 drift policies evaluate benchmark outputs (currently: `trix bench` suite v1).

## Schema (v1)

```json
{
  "policy_version": 1,
  "drift_threshold": 0.2,
  "max_churn": 0.1,
  "max_near_tie_rate": 0.3,
  "min_margin_mean": 0.001,
  "on_violation": "fallback"
}
```

Notes:
- `drift_threshold` is interpreted as a max fraction of drifted compiled classes.
- `on_violation` is reported but not automatically executed by the runtime (v1 tooling).

## Signals Used (suite v1)

From `suite_v1.json` benchmark `drift_under_regularizer_training`:
- `metrics.churn` -> `max_churn`
- `metrics.drifted_classes` + `compiled.compiled_classes` -> `max_drift_fraction`

Optional (if `telemetry.jsonl` exists and contains routing events):
- `near_tie_rate` -> `near_tie_rate_max`
- `margin_mean` -> `margin_mean_mean`

## CLI

```bash
trix drift check --suite results/benchmarks_v1/suite_v1.json --policy drift_policy.json
```

The command prints a JSON report and exits non-zero if the policy is violated.
