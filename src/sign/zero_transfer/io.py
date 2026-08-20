"""Shared save/report helpers for Phase 4 (zero-transfer) runs -- every
method's predictions already use Dataset A's exact label set
(`sarcastic`/`not_sarcastic`), so `src.classification.evaluation.metrics.compute_metrics`
is reused unmodified. This module only adds the SIGN-specific summary
numbers the brief explicitly asks for (originals-only sarcasm recall,
false-negative rate) on top of that.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.classification_settings import classification_settings
from config.sign_settings import sign_settings
from src.classification.evaluation.metrics import compute_metrics


def save_zero_transfer_result(
    experiment_id: str,
    config: dict,
    predictions: pd.DataFrame,
    family_table: pd.DataFrame,
) -> Path:
    """`predictions` must have columns `example_id, gold_label, predicted_label`
    for every SIGN test row (both roles). `family_table` is the same
    split's family table (for role lookup). Writes config/metrics/predictions
    under `results/sign/<experiment_id>/`, plus the brief-mandated
    originals-only numbers folded into metrics.json."""
    out_dir = sign_settings.results_dir / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions[["example_id", "gold_label", "predicted_label"]].to_csv(
        out_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )

    metrics = compute_metrics(
        predictions, labels=classification_settings.labels, positive_label=classification_settings.positive_label
    )

    merged = predictions.merge(family_table[["example_id", "role", "family_id"]], on="example_id", how="left")
    originals = merged[merged["role"] == "original"]
    n_originals = len(originals)
    n_correct_sarcastic = int((originals["predicted_label"] == "sarcastic").sum())
    sarcasm_recall = n_correct_sarcastic / n_originals if n_originals else float("nan")
    false_negative_rate = 1.0 - sarcasm_recall if n_originals else float("nan")

    metrics["sign_originals_summary"] = {
        "n_originals": n_originals,
        "n_correctly_predicted_sarcastic": n_correct_sarcastic,
        "sarcasm_detection_rate": sarcasm_recall,
        "false_negative_rate": false_negative_rate,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    print(
        f"[{experiment_id}] macro_f1={metrics['macro_f1']:.4f} "
        f"sarcasm_detection_rate={sarcasm_recall:.4f} "
        f"({n_correct_sarcastic}/{n_originals} originals) "
        f"false_negative_rate={false_negative_rate:.4f}"
    )
    return out_dir
