"""Shared save helper for Phase 8 (domain adaptation) runs -- unlike
Phase 4's zero-transfer (one eval target), each Phase 8 condition is
evaluated on **two** targets (SIGN Test, for RQ2/RQ3; Dataset A's own
held-out TEST, for the catastrophic-forgetting check), so results are
laid out as `results/sign/<experiment_id>/{sign_test,dataset_a_test}/`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.classification_settings import classification_settings
from config.sign_settings import sign_settings
from src.classification.evaluation.metrics import compute_metrics


def save_domain_adaptation_result(
    experiment_id: str,
    config: dict,
    sign_test_predictions: pd.DataFrame,
    dataset_a_test_predictions: pd.DataFrame,
) -> Path:
    """Both `*_predictions` frames need `example_id, gold_label,
    predicted_label`. Writes one shared `config.json` plus per-target
    `metrics.json`/`predictions.csv` under `sign_test/` and
    `dataset_a_test/` subdirectories."""
    out_dir = sign_settings.results_dir / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    targets = {
        "sign_test": sign_test_predictions,
        "dataset_a_test": dataset_a_test_predictions,
    }
    for name, predictions in targets.items():
        target_dir = out_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        cols = ["example_id", "gold_label", "predicted_label"]
        predictions[cols].to_csv(target_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        metrics = compute_metrics(
            predictions, labels=classification_settings.labels, positive_label=classification_settings.positive_label
        )
        with open(target_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[{experiment_id}/{name}] accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")

    return out_dir
