"""TF-IDF + Logistic Regression Baseline model with MLflow experiment tracking."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_latency(pipeline: Pipeline, texts: list[str]) -> tuple[float, float]:
    """Measure per-sample latency (p50 and p95) in milliseconds."""
    durations_ms = []
    # Warmup
    for text in texts[:20]:
        _ = pipeline.predict([text])

    # Benchmarking individually to simulate production requests
    for text in texts[:200]:
        t0 = time.perf_counter()
        _ = pipeline.predict([text])
        durations_ms.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(durations_ms, 50))
    p95 = float(np.percentile(durations_ms, 95))
    return p50, p95


def find_confusion_hotspots(
    y_true: list[str], y_pred: list[str], top_k: int = 10
) -> list[dict[str, Any]]:
    """Identify top confused intent pairs."""
    from collections import Counter

    confusions: Counter[tuple[str, str]] = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            confusions[(t, p)] += 1

    hotspots = [
        {"true_intent": pair[0], "predicted_intent": pair[1], "count": count}
        for pair, count in confusions.most_common(top_k)
    ]
    return hotspots


def train_baseline(
    data_dir: Path = Path("data"),
    output_model_path: Path = Path("models/baseline.joblib"),
    mlflow_tracking_uri: str | None = None,
    experiment_name: str = "ticketrouter_baseline",
) -> dict[str, Any]:
    """Train TF-IDF + Logistic Regression, evaluate on all splits, and log to MLflow."""
    if mlflow_tracking_uri is None:
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(f"Loading data from {data_dir}...")
    train_df = pd.read_json(data_dir / "train.jsonl", lines=True)
    val_df = pd.read_json(data_dir / "val.jsonl", lines=True)
    test_clean_df = pd.read_json(data_dir / "test_clean.jsonl", lines=True)
    test_shifted_df = pd.read_json(data_dir / "test_shifted.jsonl", lines=True)

    with open(data_dir / "label_names.json", encoding="utf-8") as f:
        label_names = json.load(f)

    # Convert numeric labels to string labels for end-to-end intent prediction
    y_train = train_df["label_name"].tolist()
    y_val = val_df["label_name"].tolist()
    y_test_clean = test_clean_df["label_name"].tolist()
    y_test_shifted = test_shifted_df["label_name"].tolist()

    X_train = train_df["text"].tolist()
    X_val = val_df["text"].tolist()
    X_test_clean = test_clean_df["text"].tolist()
    X_test_shifted = test_shifted_df["text"].tolist()

    # Model definition
    vectorizer_params = {
        "ngram_range": (1, 2),
        "max_features": 15000,
        "sublinear_tf": True,
    }
    classifier_params = {
        "C": 4.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": 42,
    }

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**vectorizer_params)),
            ("clf", LogisticRegression(**classifier_params)),
        ]
    )

    with mlflow.start_run(run_name="tfidf_logreg_baseline") as run:
        logger.info("Fitting TF-IDF + Logistic Regression baseline pipeline...")
        pipeline.fit(X_train, y_train)

        # Evaluate on Validation
        preds_val = pipeline.predict(X_val)
        val_acc = float(accuracy_score(y_val, preds_val))
        val_f1 = float(f1_score(y_val, preds_val, average="macro"))

        # Evaluate on Clean Test Set
        preds_clean = pipeline.predict(X_test_clean)
        clean_acc = float(accuracy_score(y_test_clean, preds_clean))
        clean_f1 = float(f1_score(y_test_clean, preds_clean, average="macro"))

        # Evaluate on Shifted Test Set
        preds_shifted = pipeline.predict(X_test_shifted)
        shifted_acc = float(accuracy_score(y_test_shifted, preds_shifted))
        shifted_f1 = float(f1_score(y_test_shifted, preds_shifted, average="macro"))

        # Latency benchmarking
        lat_p50, lat_p95 = compute_latency(pipeline, X_test_clean)

        # Confusion hot spots
        clean_hotspots = find_confusion_hotspots(y_test_clean, preds_clean)
        shifted_hotspots = find_confusion_hotspots(y_test_shifted, preds_shifted)

        # Classification reports
        clean_report = classification_report(y_test_clean, preds_clean, output_dict=True)

        # Log params and metrics to MLflow
        mlflow.log_param("model_type", "tfidf_logistic_regression")
        mlflow.log_params(vectorizer_params)
        mlflow.log_params(classifier_params)
        mlflow.log_param("num_classes", len(label_names))
        mlflow.log_param("train_samples", len(X_train))

        metrics = {
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "clean_test_accuracy": clean_acc,
            "clean_test_macro_f1": clean_f1,
            "shifted_test_accuracy": shifted_acc,
            "shifted_test_macro_f1": shifted_f1,
            "latency_p50_ms": lat_p50,
            "latency_p95_ms": lat_p95,
            "invalid_output_rate": 0.0,  # Baseline never outputs invalid strings
        }
        mlflow.log_metrics(metrics)

        # Save artifacts
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, output_model_path)
        mlflow.log_artifact(str(output_model_path), artifact_path="model")

        artifacts_dir = Path("metrics")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        hotspots_path = artifacts_dir / "baseline_confusion_hotspots.json"
        with open(hotspots_path, "w", encoding="utf-8") as f:
            json.dump(
                {"clean_hotspots": clean_hotspots, "shifted_hotspots": shifted_hotspots},
                f,
                indent=2,
            )
        mlflow.log_artifact(str(hotspots_path), artifact_path="metrics")

        report_path = artifacts_dir / "baseline_classification_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(clean_report, f, indent=2)
        mlflow.log_artifact(str(report_path), artifact_path="metrics")

        baseline_summary_path = artifacts_dir / "baseline_metrics.json"
        with open(baseline_summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            f"Baseline Run {run.info.run_id} Complete:\n"
            f"  Clean Test Acc:   {clean_acc:.4f} (F1: {clean_f1:.4f})\n"
            f"  Shifted Test Acc: {shifted_acc:.4f} (F1: {shifted_f1:.4f})\n"
            f"  Latency p50:      {lat_p50:.2f} ms | p95: {lat_p95:.2f} ms\n"
            f"  Model saved to:   {output_model_path}"
        )

        return {
            "run_id": run.info.run_id,
            "metrics": metrics,
            "model_path": str(output_model_path),
            "clean_hotspots": clean_hotspots,
        }


def main() -> None:
    """CLI entrypoint for baseline training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--output-model", type=str, default="models/baseline.joblib", help="Output model path"
    )
    parser.add_argument(
        "--tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI"
    )
    args = parser.parse_args()

    train_baseline(
        data_dir=Path(args.data_dir),
        output_model_path=Path(args.output_model),
        mlflow_tracking_uri=args.tracking_uri,
    )


if __name__ == "__main__":
    main()
