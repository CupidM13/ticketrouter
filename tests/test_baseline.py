"""Unit tests for ticketrouter.baseline module."""

from __future__ import annotations

from pathlib import Path

import joblib


def test_baseline_model_artifact_and_prediction():
    model_path = Path("models/baseline.joblib")
    assert model_path.exists(), "Baseline model joblib file must exist"

    pipeline = joblib.load(model_path)
    sample_queries = [
        "How do I unblock my PIN?",
        "I need to cancel my transfer",
        "Where is my contactless card?",
    ]
    preds = pipeline.predict(sample_queries)
    assert len(preds) == 3
    assert all(isinstance(p, str) for p in preds)

    # Test predict_proba
    probs = pipeline.predict_proba(sample_queries)
    assert probs.shape == (3, 77)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_baseline_metrics_file():
    metrics_path = Path("metrics/baseline_metrics.json")
    assert metrics_path.exists()
