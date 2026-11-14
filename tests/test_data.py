"""Unit tests for ticketrouter.data module."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from ticketrouter.data import audit_dataset, normalize_text, perturb_text


def test_normalize_text():
    raw = "  Hello, WORLD!? How are you doing?   "
    normalized = normalize_text(raw)
    assert normalized == "hello world how are you doing"


def test_perturb_text():
    rng = random.Random(42)
    raw = "Please help, my account cannot transfer money, thanks!"
    perturbed = perturb_text(raw, rng)
    assert isinstance(perturbed, str)
    # Check that slang or typo was applied
    assert "pls" in perturbed or "accnt" in perturbed or "cant" in perturbed or "thx" in perturbed


def test_audit_dataset_logic():
    label_names = ["card_lost", "transfer_issue", "pin_blocked"]
    train_df = pd.DataFrame(
        {
            "text": [
                "I lost my card",
                "Cannot transfer money",
                "My pin is blocked",
                "I lost my card please help",
                "Where is my transferred cash",
                "Help with blocked pin",
            ],
            "label": [0, 1, 2, 0, 1, 2],
        }
    )
    test_df = pd.DataFrame(
        {
            "text": [
                "I lost my card",  # Exact duplicate
                "Transfer issue query",
                "PIN blocked need reset",
            ],
            "label": [0, 1, 2],
        }
    )

    report = audit_dataset(train_df, test_df, label_names)
    assert "class_balance" in report
    assert "leakage" in report
    assert report["leakage"]["exact_leakage_count"] == 1
    assert "label_noise" in report


def test_generated_data_splits_exist():
    data_dir = Path("data")
    assert (data_dir / "train.jsonl").exists()
    assert (data_dir / "val.jsonl").exists()
    assert (data_dir / "test_clean.jsonl").exists()
    assert (data_dir / "test_shifted.jsonl").exists()
    assert (data_dir / "label_names.json").exists()
    assert (data_dir / "audit_report.json").exists()

    with open(data_dir / "label_names.json", encoding="utf-8") as f:
        labels = json.load(f)
    assert len(labels) == 77
