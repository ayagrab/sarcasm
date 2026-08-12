"""Persist / load one experiment's artifacts under `results/<experiment_id>/`.

Every experiment (every approach, every configuration) gets a unique
experiment ID and writes exactly: `config.json`, `metrics.json`,
`predictions.csv`. Nothing here computes metrics differently per approach
-- it all goes through `evaluation.metrics.compute_metrics`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.classification_settings import classification_settings
from src.classification.evaluation.metrics import compute_metrics, validate_predictions

PREDICTION_COLUMNS = ["example_id", "gold_label", "predicted_label", "confidence"]


def save_experiment_artifacts(
    experiment_id: str,
    config: dict,
    predictions: pd.DataFrame,
    extra_metrics: dict | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Validates `predictions`, computes metrics, and writes all three
    artifact files. Returns the experiment's output directory."""
    results_dir = results_dir or classification_settings.results_dir
    out_dir = results_dir / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_predictions(predictions)
    cols = [c for c in PREDICTION_COLUMNS if c in predictions.columns]
    predictions[cols].to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    metrics = compute_metrics(predictions)
    if extra_metrics:
        metrics.update(extra_metrics)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    return out_dir


def load_experiment_metrics(experiment_id: str, results_dir: Path | None = None) -> dict:
    results_dir = results_dir or classification_settings.results_dir
    path = results_dir / experiment_id / "metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_experiment_predictions(experiment_id: str, results_dir: Path | None = None) -> pd.DataFrame:
    results_dir = results_dir or classification_settings.results_dir
    path = results_dir / experiment_id / "predictions.csv"
    return pd.read_csv(path)
