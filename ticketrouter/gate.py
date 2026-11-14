"""Model Promotion Gate: Compares candidate vs production/baseline and gates registry promotion."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_gate_rules(
    candidate_metrics: dict[str, Any],
    production_metrics: dict[str, Any],
    max_latency_p95_ms: float = 800.0,
    max_invalid_output_rate: float = 0.01,
    accuracy_tolerance: float = 0.005,
    shifted_accuracy_tolerance: float = 0.010,
) -> tuple[bool, list[dict[str, Any]]]:
    """Evaluate candidate metrics against production baseline using strict promotion criteria.

    Rules:
    1. Clean Test Accuracy: candidate >= production - 0.5% point (0.005)
    2. Shifted Test Accuracy: candidate does not regress (>= production - 1.0% point)
    3. Latency: p95 latency <= configured limit
    4. Quality: invalid output rate < 1% (0.01)
    """
    checks = []

    # 1. Clean Test Accuracy
    cand_acc = candidate_metrics.get("clean_test_accuracy", candidate_metrics.get("accuracy", 0.0))
    prod_acc = production_metrics.get(
        "clean_test_accuracy", production_metrics.get("accuracy", 0.0)
    )
    min_required_acc = prod_acc - accuracy_tolerance
    acc_passed = cand_acc >= min_required_acc
    checks.append(
        {
            "rule": "clean_test_accuracy",
            "candidate_value": round(cand_acc, 4),
            "reference_value": round(prod_acc, 4),
            "threshold": f">= {round(min_required_acc, 4)} (prod - {accuracy_tolerance * 100:.1f}pt)",
            "passed": acc_passed,
        }
    )

    # 2. Shifted Test Accuracy (Drift Robustness)
    cand_shift_acc = candidate_metrics.get(
        "shifted_test_accuracy", candidate_metrics.get("accuracy", 0.0)
    )
    prod_shift_acc = production_metrics.get(
        "shifted_test_accuracy", production_metrics.get("accuracy", 0.0)
    )
    min_required_shift = prod_shift_acc - shifted_accuracy_tolerance
    shift_passed = cand_shift_acc >= min_required_shift
    checks.append(
        {
            "rule": "shifted_test_accuracy_no_regression",
            "candidate_value": round(cand_shift_acc, 4),
            "reference_value": round(prod_shift_acc, 4),
            "threshold": f">= {round(min_required_shift, 4)}",
            "passed": shift_passed,
        }
    )

    # 3. p95 Latency Check
    cand_p95 = candidate_metrics.get("latency_p95_ms", 0.0)
    latency_passed = cand_p95 <= max_latency_p95_ms
    checks.append(
        {
            "rule": "p95_latency_under_limit",
            "candidate_value": round(cand_p95, 2),
            "threshold": f"<= {max_latency_p95_ms:.1f} ms",
            "passed": latency_passed,
        }
    )

    # 4. Invalid Output Rate (< 1%)
    cand_invalid = candidate_metrics.get(
        "invalid_output_rate", candidate_metrics.get("clean_test_invalid_rate", 0.0)
    )
    invalid_passed = cand_invalid < max_invalid_output_rate
    checks.append(
        {
            "rule": "invalid_output_rate_under_1pct",
            "candidate_value": round(cand_invalid, 4),
            "threshold": f"< {max_invalid_output_rate:.2f} (1.0%)",
            "passed": invalid_passed,
        }
    )

    overall_passed = all(c["passed"] for c in checks)
    return overall_passed, checks


def promote_in_registry(
    model_name: str,
    run_id: str | None,
    client: MlflowClient,
    alias: str = "Production",
) -> None:
    """Set model alias to Production in the MLflow Model Registry."""
    try:
        # Ensure registered model exists
        try:
            client.get_registered_model(model_name)
        except Exception:
            logger.info(f"Creating registered model '{model_name}' in MLflow Registry...")
            client.create_registered_model(model_name)

        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions and run_id:
            logger.info(f"Registering model version for {model_name} from run {run_id}...")
            model_uri = f"runs:/{run_id}/lora_adapter"
            mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
            version_to_promote = mv.version
        elif versions:
            version_to_promote = max(int(v.version) for v in versions)
        else:
            mv = client.create_model_version(name=model_name, source="models/qwen_lora_adapter")
            version_to_promote = mv.version

        client.set_registered_model_alias(model_name, alias, str(version_to_promote))
        logger.info(
            f"Successfully promoted model '{model_name}' version {version_to_promote} to alias '@{alias}'."
        )
    except Exception as e:
        logger.warning(f"MLflow registry alias update note: {e}")


def run_gate(
    candidate_metrics_path: Path,
    production_metrics_path: Path,
    output_report_path: Path = Path("metrics/gate_report.json"),
    max_latency_p95_ms: float = 800.0,
    tracking_uri: str | None = None,
    model_name: str = "ticketrouter_model",
    bad_candidate: bool = False,
    promote: bool = True,
) -> bool:
    """Execute gate check, write report, and return status."""
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    mlflow.set_tracking_uri(tracking_uri)

    logger.info("Reading metrics for candidate and production...")
    with open(candidate_metrics_path, encoding="utf-8") as f:
        cand_data = json.load(f)
        cand_metrics = cand_data.get("metrics", cand_data)
        run_id = cand_data.get("run_id")

    with open(production_metrics_path, encoding="utf-8") as f:
        prod_data = json.load(f)
        prod_metrics = prod_data.get("metrics", prod_data)

    if bad_candidate:
        logger.warning("Simulating a DELIBERATELY BAD CANDIDATE to test gate rejection...")
        cand_metrics = {
            "clean_test_accuracy": 0.4200,  # Fails accuracy rule
            "shifted_test_accuracy": 0.2800,  # Fails shifted robustness
            "latency_p95_ms": 1450.0,  # Fails latency limit (> 800ms)
            "invalid_output_rate": 0.0850,  # Fails invalid rate (8.5% > 1%)
        }

    passed, checks = evaluate_gate_rules(
        candidate_metrics=cand_metrics,
        production_metrics=prod_metrics,
        max_latency_p95_ms=max_latency_p95_ms,
    )

    report = {
        "status": "PASSED" if passed else "FAILED",
        "passed": passed,
        "candidate_metrics": cand_metrics,
        "production_reference_metrics": prod_metrics,
        "checks": checks,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("================ MODEL PROMOTION GATE REPORT ================")
    logger.info(f"OVERALL STATUS: {'[PASS]' if passed else '[FAIL]'}")
    for c in checks:
        status_sym = "PASSED" if c["passed"] else "FAILED"
        logger.info(
            f" - {c['rule']:<36}: {status_sym} (candidate: {c['candidate_value']}, threshold: {c['threshold']})"
        )
    logger.info(f"Report written to: {output_report_path}")
    logger.info("=============================================================")

    if passed and promote:
        client = MlflowClient(tracking_uri=tracking_uri)
        promote_in_registry(model_name, run_id, client)

    return passed


def main() -> None:
    """CLI entrypoint for gate evaluation."""
    parser = argparse.ArgumentParser(description="TicketRouter Model Promotion Gate")
    parser.add_argument(
        "--candidate-metrics",
        type=str,
        default="metrics/candidate_metrics.json",
        help="Candidate metrics JSON",
    )
    parser.add_argument(
        "--production-metrics",
        type=str,
        default="metrics/baseline_metrics.json",
        help="Production/reference metrics JSON",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="metrics/gate_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--latency-limit-ms",
        type=float,
        default=float(os.environ.get("LATENCY_P95_LIMIT_MS", "800.0")),
        help="p95 latency limit in ms",
    )
    parser.add_argument(
        "--bad-candidate",
        action="store_true",
        help="Simulate a deliberately failing candidate to verify gate rejection",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Skip MLflow registry promotion",
    )
    args = parser.parse_args()

    passed = run_gate(
        candidate_metrics_path=Path(args.candidate_metrics),
        production_metrics_path=Path(args.production_metrics),
        output_report_path=Path(args.output_report),
        max_latency_p95_ms=args.latency_limit_ms,
        bad_candidate=args.bad_candidate,
        promote=not args.no_promote,
    )

    if not passed:
        logger.error("Promotion gate rejected the candidate model. Exiting with non-zero status.")
        sys.exit(1)

    logger.info("Promotion gate passed! Candidate is approved for Production.")
    sys.exit(0)


if __name__ == "__main__":
    main()
