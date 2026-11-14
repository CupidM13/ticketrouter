"""Unit tests for ticketrouter.monitor module."""

from __future__ import annotations

import json
from pathlib import Path

from ticketrouter.monitor import compute_psi, evaluate_drift_and_quality


def test_compute_psi_identical_distributions():
    all_classes = ["c1", "c2", "c3"]
    expected = {"c1": 100, "c2": 100, "c3": 100}
    actual = {"c1": 100, "c2": 100, "c3": 100}
    psi = compute_psi(expected, actual, all_classes)
    assert abs(psi) < 1e-5


def test_compute_psi_shifted_distributions():
    all_classes = ["c1", "c2", "c3", "c4"]
    expected = {"c1": 100, "c2": 100, "c3": 100, "c4": 100}
    # Massive shift: c1 and c2 disappear, c3 and c4 dominate
    actual = {"c1": 2, "c2": 3, "c3": 200, "c4": 195}
    psi = compute_psi(expected, actual, all_classes)
    assert psi > 0.20  # Exceeds standard drift threshold


def test_evaluate_drift_and_quality_with_live_db():
    db_path = Path("predictions.db")
    assert db_path.exists(), "predictions.db should exist after replay"

    results = evaluate_drift_and_quality(db_path=db_path)
    assert "metrics" in results
    assert "psi" in results["metrics"]
    assert "mean_confidence" in results["metrics"]
    assert "fallback_rate" in results["metrics"]


def test_retrain_request_file_generation():
    retrain_file = Path("retrain_request.json")
    assert retrain_file.exists()
    with open(retrain_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["alert_triggered"] is True
    assert len(data["reasons"]) > 0
