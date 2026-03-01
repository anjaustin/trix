import json
from pathlib import Path


def test_drift_policy_check_on_suite(tmp_path: Path):
    from trix.nn.integrity import DriftPolicy, evaluate_drift_policy_on_suite

    # Minimal fake suite structure
    suite = {
        "benchmarks": [
            {
                "name": "drift_under_regularizer_training",
                "metrics": {"churn": [0.01, 0.02, 0.03]},
            }
        ]
    }
    p = tmp_path / "suite.json"
    p.write_text(json.dumps(suite), encoding="utf-8")

    ok, rep = evaluate_drift_policy_on_suite(p, DriftPolicy(max_churn=0.05))
    assert ok
    assert rep["ok"] is True

    ok2, rep2 = evaluate_drift_policy_on_suite(p, DriftPolicy(max_churn=0.01))
    assert not ok2
    assert rep2["ok"] is False
