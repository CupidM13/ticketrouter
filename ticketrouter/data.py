"""Data processing, audit, split creation, and drift synthesis for Banking77."""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SLANG_REPLACEMENTS = {
    r"\bplease\b": "pls",
    r"\baccount\b": "accnt",
    r"\bcannot\b": "cant",
    r"\bcan't\b": "cant",
    r"\bdo not\b": "dont",
    r"\bdon't\b": "dont",
    r"\btransfer\b": "xfer",
    r"\btransferred\b": "xfered",
    r"\bworking\b": "wrkng",
    r"\bthanks\b": "thx",
    r"\bthank you\b": "thx",
    r"\btomorrow\b": "tmrw",
    r"\bapplication\b": "app",
    r"\bdeclined\b": "declnd",
    r"\bavailable\b": "avail",
    r"\bwithdraw\b": "w/draw",
    r"\bwithdrawal\b": "w/drawal",
    r"\bwithdrawing\b": "w/drawing",
    r"\btransaction\b": "tx",
    r"\btransactions\b": "txs",
    r"\bidentity\b": "id",
    r"\bidentification\b": "id",
    r"\bverification\b": "verif",
    r"\bcard\b": "crd",
    r"\bpayment\b": "pmt",
    r"\bpayments\b": "pmts",
    r"\bbalance\b": "bal",
    r"\bdeposit\b": "dpst",
}


def normalize_text(text: str) -> str:
    """Normalize text for near-duplicate detection."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def perturb_text(text: str, rng: random.Random) -> str:
    """Apply realistic typos, lowercase slang, missing apostrophes, and abbreviations."""
    perturbed = text.lower()

    # Apply slang replacements probabilistically
    for pattern, replacement in SLANG_REPLACEMENTS.items():
        if re.search(pattern, perturbed, flags=re.IGNORECASE) and rng.random() < 0.75:
            perturbed = re.sub(pattern, replacement, perturbed, flags=re.IGNORECASE)

    # Remove apostrophes
    perturbed = perturbed.replace("'", "").replace("’", "")

    # Inject minor character typos (adjacent swap, deletion, or double letter)
    words = perturbed.split()
    new_words = []
    for word in words:
        if len(word) > 4 and rng.random() < 0.35:
            typo_type = rng.choice(["swap", "drop", "double"])
            idx = rng.randint(1, len(word) - 2)
            if typo_type == "swap":
                word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :]
            elif typo_type == "drop":
                word = word[:idx] + word[idx + 1 :]
            elif typo_type == "double":
                word = word[:idx] + word[idx] + word[idx:]
        new_words.append(word)

    return " ".join(new_words)


def audit_dataset(
    train_df: pd.DataFrame, test_df: pd.DataFrame, label_names: list[str]
) -> dict[str, Any]:
    """Audit class balance, data leakage across splits, and likely label noise."""
    logger.info("Auditing Banking77 dataset...")

    # 1. Class Balance Audit
    train_counts = train_df["label"].value_counts().sort_index().to_dict()
    test_counts = test_df["label"].value_counts().sort_index().to_dict()

    class_balance = {
        "num_classes": len(label_names),
        "train_min_samples": int(min(train_counts.values())),
        "train_max_samples": int(max(train_counts.values())),
        "train_median_samples": float(np.median(list(train_counts.values()))),
        "test_min_samples": int(min(test_counts.values())),
        "test_max_samples": int(max(test_counts.values())),
    }

    # 2. Leakage Audit (Exact and Near Duplicates)
    train_exact = set(train_df["text"])
    exact_leakage = [row for row in test_df.to_dict(orient="records") if row["text"] in train_exact]

    train_norm_map: dict[str, int] = {
        normalize_text(t): int(lbl) for t, lbl in zip(train_df["text"], train_df["label"])
    }
    near_leakage = []
    for row in test_df.to_dict(orient="records"):
        norm = normalize_text(row["text"])
        if norm in train_norm_map:
            near_leakage.append(
                {
                    "test_text": row["text"],
                    "test_label": label_names[row["label"]],
                    "train_label": label_names[train_norm_map[norm]],
                    "label_matches": bool(row["label"] == train_norm_map[norm]),
                }
            )

    logger.info(
        f"Found {len(exact_leakage)} exact leakages and {len(near_leakage)} normalized near-leakages."
    )

    # 3. Label Noise Detection via Stratified Cross-Validation
    logger.info("Running cross-validation to detect likely label noise in training set...")
    texts = train_df["text"].tolist()
    labels = np.array(train_df["label"].tolist())

    min_class_samples = int(min(train_counts.values())) if train_counts else 0
    n_splits = max(2, min(5, min_class_samples)) if min_class_samples >= 2 else 0

    label_noise_candidates = []
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(labels), dtype=int)
        oof_probs = np.zeros(len(labels), dtype=float)

        for train_idx, val_idx in skf.split(texts, labels):
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
            X_tr = vec.fit_transform([texts[i] for i in train_idx])
            y_tr = labels[train_idx]

            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(X_tr, y_tr)

            X_val = vec.transform([texts[i] for i in val_idx])
            probs = clf.predict_proba(X_val)
            preds = np.argmax(probs, axis=1)

            oof_preds[val_idx] = preds
            oof_probs[val_idx] = np.max(probs, axis=1)

        for idx, (true_lbl, pred_lbl, prob) in enumerate(zip(labels, oof_preds, oof_probs)):
            if true_lbl != pred_lbl and prob >= 0.60:
                label_noise_candidates.append(
                    {
                        "index": int(idx),
                        "text": texts[idx],
                        "true_label": label_names[true_lbl],
                        "predicted_label": label_names[pred_lbl],
                        "confidence": float(prob),
                    }
                )

        label_noise_candidates.sort(key=lambda x: x["confidence"], reverse=True)
        logger.info(
            f"Identified {len(label_noise_candidates)} likely label-noise training samples (CV conf >= 0.60)."
        )

    report = {
        "class_balance": class_balance,
        "leakage": {
            "exact_leakage_count": len(exact_leakage),
            "near_leakage_count": len(near_leakage),
            "sample_near_leakages": near_leakage[:5],
        },
        "label_noise": {
            "num_candidates": len(label_noise_candidates),
            "top_candidates": label_noise_candidates[:10],
        },
    }
    return report


def load_and_prepare_data(
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Load Banking77, audit, remove leakage, create stratified train/val/test and drifted test set."""
    ds = load_dataset("PolyAI/banking77")
    label_names: list[str] = ds["train"].features["label"].names

    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])

    # Run full audit
    audit_results = audit_dataset(train_df, test_df, label_names)

    # 1. Remove Leakage from Test Set
    train_norm_set = {normalize_text(t) for t in train_df["text"]}
    clean_test_mask = [normalize_text(t) not in train_norm_set for t in test_df["text"]]
    clean_test_df = test_df[clean_test_mask].copy().reset_index(drop=True)
    clean_test_df["label_name"] = [label_names[lbl] for lbl in clean_test_df["label"]]

    logger.info(
        f"Removed {len(test_df) - len(clean_test_df)} leaking queries from test set. Clean test size: {len(clean_test_df)}"
    )

    # 2. Stratified Train / Val split from training data
    train_split_df, val_split_df = train_test_split(
        train_df,
        test_size=val_ratio,
        stratify=train_df["label"],
        random_state=seed,
    )
    train_split_df = train_split_df.copy().reset_index(drop=True)
    val_split_df = val_split_df.copy().reset_index(drop=True)

    train_split_df["label_name"] = [label_names[lbl] for lbl in train_split_df["label"]]
    val_split_df["label_name"] = [label_names[lbl] for lbl in val_split_df["label"]]

    # 3. Create Drift Scenario: "shifted" test set
    # 15 held-out intents become rare (85% reduction)
    # Remaining 62 intents represent the primary traffic
    rng = random.Random(seed)
    all_label_indices = list(range(len(label_names)))
    rng.shuffle(all_label_indices)
    held_out_15_intents = set(all_label_indices[:15])

    shifted_rows = []
    for _, row in clean_test_df.iterrows():
        lbl = int(row["label"])
        # If in held-out 15 intents, downsample heavily
        if lbl in held_out_15_intents and rng.random() > 0.15:
            continue

        # Perturb query text with typos, slang, missing apostrophes
        p_text = perturb_text(row["text"], rng)
        shifted_rows.append(
            {
                "text": p_text,
                "original_text": row["text"],
                "label": lbl,
                "label_name": label_names[lbl],
                "is_held_out_intent": bool(lbl in held_out_15_intents),
            }
        )

    # Oversample the boosted intents to maintain a realistic traffic volume (~3000 requests)
    boosted_rows = [r for r in shifted_rows if not r["is_held_out_intent"]]
    target_count = len(clean_test_df)
    while len(shifted_rows) < target_count:
        sampled_row = dict(rng.choice(boosted_rows))
        sampled_row["text"] = perturb_text(sampled_row["original_text"], rng)
        shifted_rows.append(sampled_row)

    shifted_test_df = pd.DataFrame(shifted_rows).reset_index(drop=True)
    logger.info(
        f"Created shifted test set with {len(shifted_test_df)} samples, "
        f"15 down-sampled intents: {[label_names[i] for i in sorted(held_out_15_intents)]}"
    )

    audit_results["drift_config"] = {
        "held_out_15_intents": [label_names[i] for i in sorted(held_out_15_intents)],
        "shifted_test_samples": len(shifted_test_df),
        "clean_test_samples": len(clean_test_df),
        "train_samples": len(train_split_df),
        "val_samples": len(val_split_df),
    }

    return train_split_df, val_split_df, clean_test_df, shifted_test_df, label_names, audit_results


def save_splits(
    output_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_clean_df: pd.DataFrame,
    test_shifted_df: pd.DataFrame,
    label_names: list[str],
    audit_report: dict[str, Any],
) -> None:
    """Save processed datasets and metadata to JSONL/JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_json(output_dir / "train.jsonl", orient="records", lines=True)
    val_df.to_json(output_dir / "val.jsonl", orient="records", lines=True)
    test_clean_df.to_json(output_dir / "test_clean.jsonl", orient="records", lines=True)
    test_shifted_df.to_json(output_dir / "test_shifted.jsonl", orient="records", lines=True)

    with open(output_dir / "label_names.json", "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2)

    with open(output_dir / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    logger.info(f"All datasets and audit reports saved successfully to {output_dir}")


def main() -> None:
    """CLI entrypoint for data preparation."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Banking77 dataset splits and audit.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_df, val_df, test_clean, test_shifted, label_names, audit = load_and_prepare_data(
        val_ratio=args.val_ratio, seed=args.seed
    )
    save_splits(
        Path(args.output_dir), train_df, val_df, test_clean, test_shifted, label_names, audit
    )


if __name__ == "__main__":
    main()
