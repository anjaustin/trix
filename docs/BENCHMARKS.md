# Benchmarks

This repo includes a small, CPU-first benchmark suite intended to make core claims falsifiable.

## Suite v1

Run:

```bash
python -m pip install -e ".[dev]"
python experiments/benchmarks/benchmark_suite_v1.py --outdir results/benchmarks_v1 --device cpu
```

Artifacts:
- `results/benchmarks_v1/suite_v1.json` : summary (pass/fail + metrics)
- `results/benchmarks_v1/routing_telemetry.jsonl` : routing/stability telemetry stream

Notes:
- `--include-slow` enables an 8-bit adder exhaustive check (65,536 cases). It is optional.

## What This Measures

- FP4 atoms correctness on truth tables (exactness-by-construction lane)
- Routing specialization + CompiledDispatch semantics on a small synthetic task (routing-as-primitive lane)
- Stability probe: churn under controlled signature perturbations (bounded, measurable)
- Drift under optimization: routing churn and compiled contract drift under regularizer-driven signature updates
