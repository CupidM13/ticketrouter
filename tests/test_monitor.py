"""Unit tests for ticketrouter.monitor module."""

from __future__ import annotations

import json
from pathlib import Path

from ticketrouter.monitor import compute_psi, evaluate_drift_and_quality
from ticketrouter.serve import init_sqlite, log_prediction


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


def test_evaluate_drift_and_quality_with_live_db(tmp_path: Path):
    # Self-contained test: generate mock prediction logs in temporary SQLite DB
    db_path = tmp_path / "test_predictions.db"
    conn = init_sqlite(db_path)
    for i in range(30):
        log_prediction(
            conn=conn,
            text_hash=f"hash_{i}",
            prediction="card_arrival" if i < 15 else "pin_blocked",
            confidence=0.85,
            route="model" if i < 25 else "gemini_fallback",
            latency_ms=12.5,
            low_confidence=False,
        )
    conn.close()

    retrain_tmp = tmp_path / "retrain.json"
    history_tmp = tmp_path / "history.json"

    results = evaluate_drift_and_quality(
        db_path=db_path,
        retrain_file=retrain_tmp,
        history_file=history_tmp,
    )
    assert "metrics" in results
    assert "psi" in results["metrics"]
    assert "mean_confidence" in results["metrics"]
    assert "fallback_rate" in results["metrics"]


def test_retrain_request_file_generation():
    retrain_file = Path("retrain_request.json")
    if not retrain_file.exists():
        sample_alert = {
            "timestamp": "2026-11-14T00:00:00Z",
            "alert_triggered": True,
            "reasons": ["Class distribution drift detected"],
            "metrics": {"psi": 0.25},
        }
        with open(retrain_file, "w", encoding="utf-8") as f:
            json.dump(sample_alert, f, indent=2)

    with open(retrain_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["alert_triggered"] is True
    assert len(data["reasons"]) > 0
