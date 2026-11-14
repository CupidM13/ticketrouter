"""Unit tests for ticketrouter.gate module."""

from __future__ import annotations

from pathlib import Path

from ticketrouter.gate import evaluate_gate_rules, run_gate


def test_evaluate_gate_rules_pass():
    candidate = {
        "clean_test_accuracy": 0.915,
        "shifted_test_accuracy": 0.840,
        "latency_p95_ms": 320.0,
        "invalid_output_rate": 0.003,
    }
    production = {
        "clean_test_accuracy": 0.880,
        "shifted_test_accuracy": 0.720,
        "latency_p95_ms": 350.0,
        "invalid_output_rate": 0.0,
    }

    passed, checks = evaluate_gate_rules(candidate, production, max_latency_p95_ms=800.0)
    assert passed is True
    assert all(c["passed"] for c in checks)


def test_evaluate_gate_rules_fail_on_accuracy_drop():
    candidate = {
        "clean_test_accuracy": 0.850,  # Below prod 0.880 - 0.005 = 0.875
        "shifted_test_accuracy": 0.840,
        "latency_p95_ms": 200.0,
        "invalid_output_rate": 0.001,
    }
    production = {
        "clean_test_accuracy": 0.880,
        "shifted_test_accuracy": 0.720,
    }
    passed, checks = evaluate_gate_rules(candidate, production)
    assert passed is False
    acc_check = next(c for c in checks if c["rule"] == "clean_test_accuracy")
    assert acc_check["passed"] is False


def test_evaluate_gate_rules_fail_on_invalid_output_rate():
    candidate = {
        "clean_test_accuracy": 0.920,
        "shifted_test_accuracy": 0.850,
        "latency_p95_ms": 200.0,
        "invalid_output_rate": 0.025,  # 2.5% > 1.0%
    }
    production = {
        "clean_test_accuracy": 0.880,
        "shifted_test_accuracy": 0.720,
    }
    passed, checks = evaluate_gate_rules(candidate, production)
    assert passed is False
    inv_check = next(c for c in checks if c["rule"] == "invalid_output_rate_under_1pct")
    assert inv_check["passed"] is False


def test_run_gate_bad_candidate_returns_false(tmp_path: Path):
    cand_path = Path("metrics/candidate_metrics.json")
    prod_path = Path("metrics/production_metrics.json")
    report_path = tmp_path / "gate_report.json"

    passed = run_gate(
        candidate_metrics_path=cand_path,
        production_metrics_path=prod_path,
        output_report_path=report_path,
        bad_candidate=True,
        promote=False,
    )
    assert passed is False
    assert report_path.exists()
