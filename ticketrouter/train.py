"""LoRA Supervised Fine-Tuning (SFT) for intent classification using TRL."""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Prevent incompatible torchao (< 0.16.0) preinstalled in Kaggle from breaking PEFT dispatchers
try:
    import torchao
    from packaging import version

    if version.parse(getattr(torchao, "__version__", "0.0.0")) < version.parse("0.16.0"):
        sys.modules["torchao"] = None
except Exception:
    pass

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an enterprise banking query intent classifier. "
    "Classify the customer inquiry into exactly one of the 77 standard intents. "
    "Output ONLY the exact intent label."
)


def format_instruction(text: str, label_name: str | None = None) -> str:
    """Format prompt with ChatML style for Qwen models."""
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nQuery: {text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if label_name is not None:
        return prompt + f"{label_name}<|im_end|>"
    return prompt


def map_to_nearest_label(
    raw_output: str, valid_labels: list[str], valid_labels_set: set[str]
) -> tuple[str, bool]:
    """Map raw model output to nearest valid intent. Returns (mapped_label, is_invalid_raw)."""
    # Clean output: first line, lowercased, stripped punctuation
    cleaned = raw_output.strip().split("\n")[0].strip().lower()
    cleaned = cleaned.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if cleaned.startswith("intent:"):
        cleaned = cleaned[7:].strip()

    if cleaned in valid_labels_set:
        return cleaned, False

    matches = difflib.get_close_matches(cleaned, valid_labels, n=1, cutoff=0.0)
    nearest = matches[0] if matches else valid_labels[0]
    return nearest, True


class MLflowLossCallback(TrainerCallback):
    """Custom callback to stream step-level training loss to MLflow."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and mlflow.active_run():
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=state.global_step)


def evaluate_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    eval_df: pd.DataFrame,
    valid_labels: list[str],
    device: torch.device,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate accuracy, macro-F1, latency, and invalid-output rate on a dataset split."""
    if max_eval_samples and len(eval_df) > max_eval_samples:
        eval_df = eval_df.iloc[:max_eval_samples].copy()

    valid_labels_set = set(valid_labels)
    model.eval()

    y_true = eval_df["label_name"].tolist()
    y_pred = []
    latencies_ms = []
    invalid_count = 0

    with torch.no_grad():
        for text in eval_df["text"].tolist():
            prompt = format_instruction(text)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            t0 = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            mapped_lbl, is_invalid = map_to_nearest_label(raw_text, valid_labels, valid_labels_set)
            if is_invalid:
                invalid_count += 1
            y_pred.append(mapped_lbl)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    inv_rate = float(invalid_count / max(1, len(y_true)))
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "invalid_output_rate": inv_rate,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "samples_evaluated": len(y_true),
    }


def train_lora(
    base_model_name: str | None = None,
    data_dir: Path = Path("data"),
    output_dir: Path = Path("models/qwen_lora_adapter"),
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    smoke: bool = False,
    tracking_uri: str | None = None,
    experiment_name: str = "ticketrouter_sft",
    registered_model_name: str = "ticketrouter_model",
) -> dict[str, Any]:
    """Train LoRA fine-tuned model with TRL SFTTrainer and log everything to MLflow."""
    if base_model_name is None:
        base_model_name = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Determine Device & Precision
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()
    if is_cuda:
        device = torch.device("cuda")
        torch_dtype = torch.float16  # T4-friendly fp16
        use_fp16 = True
    elif is_mps and not smoke:
        device = torch.device("mps")
        torch_dtype = torch.float32
        use_fp16 = False
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32
        use_fp16 = False

    logger.info(
        f"Training configuration: device={device}, dtype={torch_dtype}, "
        f"use_fp16={use_fp16}, smoke={smoke}, base_model={base_model_name}"
    )

    # Load splits
    train_df = pd.read_json(data_dir / "train.jsonl", lines=True)
    val_df = pd.read_json(data_dir / "val.jsonl", lines=True)
    test_clean_df = pd.read_json(data_dir / "test_clean.jsonl", lines=True)
    test_shifted_df = pd.read_json(data_dir / "test_shifted.jsonl", lines=True)

    with open(data_dir / "label_names.json", encoding="utf-8") as f:
        valid_labels = json.load(f)

    if smoke:
        logger.info("SMOKE MODE: subsampling datasets (200 train, 40 val, 40 test)")
        train_df = train_df.iloc[:200].copy()
        val_df = val_df.iloc[:40].copy()
        test_clean_df = test_clean_df.iloc[:40].copy()
        test_shifted_df = test_shifted_df.iloc[:40].copy()

    # Create formatted text column
    train_df["formatted_text"] = [
        format_instruction(t, lbl) for t, lbl in zip(train_df["text"], train_df["label_name"])
    ]
    val_df["formatted_text"] = [
        format_instruction(t, lbl) for t, lbl in zip(val_df["text"], val_df["label_name"])
    ]

    train_ds = Dataset.from_pandas(train_df[["formatted_text"]])
    val_ds = Dataset.from_pandas(val_df[["formatted_text"]])

    # Load Tokenizer
    logger.info(f"Loading tokenizer: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Base Model
    logger.info(f"Loading base model: {base_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=None if (smoke or not is_cuda) else "auto",
        trust_remote_code=True,
    )
    if smoke or not is_cuda:
        model.to(device)

    # Configure LoRA: r=16, alpha=32, target all linear projections
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    max_steps = 20 if smoke else -1
    effective_epochs = 1 if smoke else epochs
    grad_accum = 1 if smoke else 2
    grad_checkpointing = not smoke and is_cuda

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        dataset_text_field="formatted_text",
        max_length=256,
        num_train_epochs=effective_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        fp16=use_fp16,
        logging_steps=5 if smoke else 20,
        eval_strategy="no" if smoke else "epoch",
        save_strategy="no" if smoke else "epoch",
        gradient_checkpointing=grad_checkpointing,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if not smoke else None,
        callbacks=[MLflowLossCallback()],
    )

    with mlflow.start_run(run_name="qwen_lora_sft") as run:
        mlflow.log_params(
            {
                "base_model": base_model_name,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "learning_rate": learning_rate,
                "epochs": effective_epochs,
                "batch_size": batch_size,
                "smoke": smoke,
                "device": str(device),
                "fp16": use_fp16,
            }
        )

        logger.info("Starting SFT training...")
        train_result = trainer.train()
        mlflow.log_metrics(
            {
                "train_loss": train_result.training_loss,
                "train_runtime_sec": train_result.metrics.get("train_runtime", 0.0),
            }
        )

        # Save LoRA Adapter
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        mlflow.log_artifact(str(output_dir), artifact_path="lora_adapter")
        logger.info(f"LoRA adapter saved to {output_dir}")

        # Post-training evaluation
        logger.info("Evaluating fine-tuned model on clean test set...")
        clean_eval = evaluate_model(
            model,
            tokenizer,
            test_clean_df,
            valid_labels,
            device,
            max_eval_samples=40 if smoke else 500,
        )

        logger.info("Evaluating fine-tuned model on shifted test set...")
        shifted_eval = evaluate_model(
            model,
            tokenizer,
            test_shifted_df,
            valid_labels,
            device,
            max_eval_samples=40 if smoke else 500,
        )

        metrics = {
            "clean_test_accuracy": clean_eval["accuracy"],
            "clean_test_macro_f1": clean_eval["macro_f1"],
            "clean_test_invalid_rate": clean_eval["invalid_output_rate"],
            "shifted_test_accuracy": shifted_eval["accuracy"],
            "shifted_test_macro_f1": shifted_eval["macro_f1"],
            "latency_p50_ms": clean_eval["latency_p50_ms"],
            "latency_p95_ms": clean_eval["latency_p95_ms"],
        }
        mlflow.log_metrics(metrics)

        # Save metrics JSON artifact for gate.py
        metrics_file = Path("metrics/candidate_metrics.json")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run.info.run_id,
                    "model_name": registered_model_name,
                    "metrics": metrics,
                    "clean_eval": clean_eval,
                    "shifted_eval": shifted_eval,
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(str(metrics_file), artifact_path="metrics")

        logger.info(
            f"Fine-Tuning Run {run.info.run_id} Completed:\n"
            f"  Clean Test Acc:   {clean_eval['accuracy']:.4f} (F1: {clean_eval['macro_f1']:.4f})\n"
            f"  Shifted Test Acc: {shifted_eval['accuracy']:.4f} (F1: {shifted_eval['macro_f1']:.4f})\n"
            f"  Invalid Output:   {clean_eval['invalid_output_rate']:.4f}\n"
            f"  Latency p95:      {clean_eval['latency_p95_ms']:.2f} ms"
        )

        return {
            "run_id": run.info.run_id,
            "metrics": metrics,
            "adapter_dir": str(output_dir),
        }


def main() -> None:
    """CLI entrypoint for LoRA SFT."""
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for Banking77 Intent Routing")
    parser.add_argument("--base-model", type=str, default=None, help="Base model name or path")
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory with dataset splits"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/qwen_lora_adapter", help="Output adapter dir"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--smoke", action="store_true", help="Run 20-step smoke test on 200 samples"
    )
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI")
    args = parser.parse_args()

    train_lora(
        base_model_name=args.base_model,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        smoke=args.smoke,
        tracking_uri=args.tracking_uri,
    )


if __name__ == "__main__":
    main()
