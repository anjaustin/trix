import json
from pathlib import Path


def _write_suite(path: Path, telemetry_jsonl: str) -> None:
    suite = {
        "suite": "trix_benchmark_v1",
        "run_id": "test",
        "ok": True,
        "benchmarks": [
            {
                "name": "drift_under_regularizer_training",
                "ok": True,
                "config": {"num_classes": 2},
                "compiled": {"compiled_classes": [0, 1]},
                "metrics": {
                    "churn": [0.0, 0.0],
                    "drifted_classes": [[], []],
                    "compiled_hit_rate": [1.0, 1.0],
                },
                "telemetry": {"jsonl": telemetry_jsonl},
            }
        ],
    }
    path.write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_drift_eval_missing_telemetry_jsonl_warns(tmp_path: Path):
    from trix.nn.integrity import DriftPolicy, evaluate_drift_policy_on_suite

    suite_path = tmp_path / "suite_v1.json"
    missing = str(tmp_path / "does_not_exist.jsonl")
    _write_suite(suite_path, missing)

    ok, rep = evaluate_drift_policy_on_suite(suite_path, DriftPolicy())
    assert ok is True
    assert rep.get("warnings")
    assert any("missing" in w for w in rep["warnings"])


def test_drift_eval_corrupt_telemetry_jsonl_warns(tmp_path: Path):
    from trix.nn.integrity import DriftPolicy, evaluate_drift_policy_on_suite

    telem = tmp_path / "drift_telemetry.jsonl"
    telem.write_text("{not json}\n", encoding="utf-8")

    suite_path = tmp_path / "suite_v1.json"
    _write_suite(suite_path, str(telem))

    ok, rep = evaluate_drift_policy_on_suite(suite_path, DriftPolicy())
    assert ok is True
    assert rep.get("warnings")
