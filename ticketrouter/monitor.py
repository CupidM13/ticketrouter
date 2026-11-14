"""Drift and Quality Monitoring over the prediction log with retrain triggering."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default Thresholds
PSI_THRESHOLD = float(os.environ.get("DRIFT_PSI_THRESHOLD", "0.20"))
FALLBACK_THRESHOLD = float(os.environ.get("DRIFT_FALLBACK_THRESHOLD", "0.35"))
CONFIDENCE_DROP_THRESHOLD = float(os.environ.get("DRIFT_CONFIDENCE_DROP", "0.15"))
DEFAULT_WINDOW_SIZE = int(os.environ.get("DRIFT_WINDOW_SIZE", "150"))
RETRAIN_REQUEST_FILE = Path("retrain_request.json")
ALERT_HISTORY_FILE = Path("metrics/alert_history.json")


def compute_psi(
    expected_counts: dict[str, int],
    actual_counts: dict[str, int],
    all_classes: list[str],
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and live distribution."""
    total_expected = sum(expected_counts.values())
    total_actual = sum(actual_counts.values())

    if total_expected == 0 or total_actual == 0:
        return 0.0

    psi = 0.0
    k = len(all_classes)
    for cls in all_classes:
        # Laplace smoothed proportions
        exp_p = (expected_counts.get(cls, 0) + epsilon) / (total_expected + k * epsilon)
        act_p = (actual_counts.get(cls, 0) + epsilon) / (total_actual + k * epsilon)

        term = (act_p - exp_p) * math.log(act_p / exp_p)
        psi += term

    return float(psi)


def get_training_intent_distribution(
    data_dir: Path = Path("data"),
) -> tuple[dict[str, int], list[str]]:
    """Compute intent frequencies from the training split."""
    train_file = data_dir / "train.jsonl"
    labels_file = data_dir / "label_names.json"

    with open(labels_file, encoding="utf-8") as f:
        all_labels = json.load(f)

    train_df = pd.read_json(train_file, lines=True)
    counts = train_df["label_name"].value_counts().to_dict()
    return counts, all_labels


def fetch_recent_predictions(db_path: Path, window_size: int = DEFAULT_WINDOW_SIZE) -> pd.DataFrame:
    """Fetch the latest window_size predictions from SQLite."""
    if not db_path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT timestamp, text_hash, prediction, confidence, route, latency_ms, low_confidence
        FROM predictions
        ORDER BY id DESC
        LIMIT {window_size}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def evaluate_drift_and_quality(
    db_path: Path = Path("predictions.db"),
    data_dir: Path = Path("data"),
    window_size: int = DEFAULT_WINDOW_SIZE,
    psi_threshold: float = PSI_THRESHOLD,
    fallback_threshold: float = FALLBACK_THRESHOLD,
    confidence_drop_threshold: float = CONFIDENCE_DROP_THRESHOLD,
    retrain_file: Path | None = None,
    history_file: Path | None = None,
) -> dict[str, Any]:
    """Analyze the recent prediction window for drift and quality degradation."""
    df = fetch_recent_predictions(db_path, window_size)
    if len(df) < 20:
        logger.info(f"Insufficient predictions ({len(df)} < 20) for statistical drift analysis.")
        return {"status": "insufficient_data", "samples": len(df)}

    train_counts, all_labels = get_training_intent_distribution(data_dir)

    # 1. Class Distribution Shift (PSI)
    live_counts = df["prediction"].value_counts().to_dict()
    psi_val = compute_psi(train_counts, live_counts, all_labels)

    # 2. Confidence Distribution
    mean_conf = float(df["confidence"].mean())
    baseline_conf = 0.90  # expected baseline confidence
    conf_drop = max(0.0, baseline_conf - mean_conf)

    # 3. Fallback Rate
    fallback_count = int((df["route"] != "model").sum())
    fallback_rate = float(fallback_count / len(df))

    # 4. Latency
    p50_lat = float(df["latency_ms"].median())
    p95_lat = float(df["latency_ms"].quantile(0.95))

    # Evaluate Alerts
    reasons = []
    if psi_val >= psi_threshold:
        reasons.append(
            f"Class distribution drift detected: PSI = {psi_val:.4f} >= threshold ({psi_threshold:.2f})"
        )
    if fallback_rate >= fallback_threshold:
        reasons.append(
            f"High fallback rate detected: {fallback_rate * 100:.1f}% >= threshold ({fallback_threshold * 100:.1f}%)"
        )
    if conf_drop >= confidence_drop_threshold:
        reasons.append(
            f"Confidence drop detected: mean confidence {mean_conf:.3f} (drop {conf_drop:.3f} >= {confidence_drop_threshold:.2f})"
        )

    alert_triggered = len(reasons) > 0

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window_size": len(df),
        "alert_triggered": alert_triggered,
        "reasons": reasons,
        "metrics": {
            "psi": round(psi_val, 4),
            "mean_confidence": round(mean_conf, 4),
            "confidence_drop": round(conf_drop, 4),
            "fallback_rate": round(fallback_rate, 4),
            "fallback_count": fallback_count,
            "latency_p50_ms": round(p50_lat, 2),
            "latency_p95_ms": round(p95_lat, 2),
        },
        "thresholds": {
            "psi_threshold": psi_threshold,
            "fallback_threshold": fallback_threshold,
            "confidence_drop_threshold": confidence_drop_threshold,
        },
    }

    retrain_target = retrain_file or RETRAIN_REQUEST_FILE
    history_target = history_file or ALERT_HISTORY_FILE

    if alert_triggered:
        logger.warning("DRIFT ALERT TRIGGERED! Reasons:\n" + "\n".join(f" - {r}" for r in reasons))
        # Write retrain_request.json
        retrain_target.parent.mkdir(parents=True, exist_ok=True)
        with open(retrain_target, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Retrain request written to {retrain_target}")

        # Append to alert history
        append_alert_history(results, history_file=history_target)
    else:
        logger.info(
            f"Monitoring Check OK: PSI={psi_val:.4f}, Fallback={fallback_rate * 100:.1f}%, MeanConf={mean_conf:.3f}"
        )

    return results


def append_alert_history(
    alert_data: dict[str, Any], history_file: Path = ALERT_HISTORY_FILE
) -> None:
    """Append alert event to history log."""
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if history_file.exists():
        try:
            with open(history_file, encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(alert_data)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def replay_traffic(
    service_url: str = "http://127.0.0.1:8000/predict",
    data_dir: Path = Path("data"),
    num_samples_per_split: int = 150,
) -> None:
    """Replay clean traffic then shifted traffic to demonstrate drift detection."""
    logger.info("Starting live traffic replay simulation...")
    clean_file = data_dir / "test_clean.jsonl"
    shifted_file = data_dir / "test_shifted.jsonl"

    clean_df = (
        pd.read_json(clean_file, lines=True)
        .sample(frac=1.0, random_state=42)
        .iloc[:num_samples_per_split]
    )
    shifted_df = pd.read_json(shifted_file, lines=True).iloc[:num_samples_per_split]

    # Phase 1: Replay Clean Traffic
    logger.info(f"--- Phase 1: Replaying {len(clean_df)} clean queries ---")
    for idx, text in enumerate(clean_df["text"]):
        try:
            requests.post(service_url, json={"text": text}, timeout=5.0)
        except Exception:
            # Fallback to in-process execution if server is offline
            from ticketrouter.serve import app, predict_intent, state

            if state.pipeline is None:
                # initialize in process
                import asyncio

                from ticketrouter.serve import lifespan

                asyncio.run(lifespan(app).__aenter__())
            predict_intent(text, state.threshold)

        if (idx + 1) % 50 == 0:
            logger.info(f"Clean replay progress: {idx + 1}/{len(clean_df)}")

    # Check drift after clean replay
    logger.info("Evaluating drift after clean traffic...")
    clean_status = evaluate_drift_and_quality()
    logger.info(f"Clean traffic result: Alert={clean_status.get('alert_triggered', False)}")

    # Phase 2: Replay Shifted Traffic
    logger.info(f"--- Phase 2: Replaying {len(shifted_df)} drifted/shifted queries ---")
    for idx, text in enumerate(shifted_df["text"]):
        try:
            requests.post(service_url, json={"text": text}, timeout=5.0)
        except Exception:
            from ticketrouter.serve import predict_intent, state

            predict_intent(text, state.threshold)

        if (idx + 1) % 50 == 0:
            logger.info(f"Shifted replay progress: {idx + 1}/{len(shifted_df)}")

    # Check drift after shifted replay
    logger.info("Evaluating drift after shifted traffic...")
    shifted_status = evaluate_drift_and_quality()
    logger.info(
        f"Shifted traffic result: Alert={shifted_status.get('alert_triggered', False)}, "
        f"PSI={shifted_status.get('metrics', {}).get('psi')}"
    )


def main() -> None:
    """CLI entrypoint for drift monitoring and replay."""
    parser = argparse.ArgumentParser(description="TicketRouter Drift Monitor")
    parser.add_argument("--replay", action="store_true", help="Replay clean and shifted traffic")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_SIZE, help="Window size")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/predict")
    args = parser.parse_args()

    if args.replay:
        replay_traffic(service_url=args.url, num_samples_per_split=args.window)
    else:
        evaluate_drift_and_quality(window_size=args.window)


if __name__ == "__main__":
    main()
