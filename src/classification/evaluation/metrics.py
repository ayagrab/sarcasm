"""Shared evaluation logic, used identically by every approach (M1-M6).

Every approach must produce predictions in one normalized schema:

    {"example_id": ..., "gold_label": ..., "predicted_label": ..., "confidence": optional}

so results are directly comparable across approaches. `compute_metrics`
is the single implementation of every metric used for model selection and
final comparison -- no approach computes its own metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from config.classification_settings import classification_settings

REQUIRED_PREDICTION_COLUMNS = {"example_id", "gold_label", "predicted_label"}


def validate_predictions(df: pd.DataFrame, valid_labels: tuple[str, ...] | None = None) -> None:
    valid_labels = valid_labels or classification_settings.labels
    missing_cols = REQUIRED_PREDICTION_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Predictions dataframe missing columns: {missing_cols}")
    if df["example_id"].duplicated().any():
        dupes = df.loc[df["example_id"].duplicated(), "example_id"].tolist()
        raise ValueError(f"Duplicate example_id in predictions: {dupes}")
    invalid_gold = set(df["gold_label"].unique()) - set(valid_labels)
    if invalid_gold:
        raise ValueError(f"Unexpected gold_label values: {invalid_gold}")
    invalid_pred = set(df["predicted_label"].unique()) - set(valid_labels)
    if invalid_pred:
        raise ValueError(
            f"Unexpected predicted_label values (output not normalized?): {invalid_pred}"
        )


def compute_metrics(
    df: pd.DataFrame,
    labels: tuple[str, ...] | None = None,
    positive_label: str | None = None,
) -> dict:
    """Accuracy, per-class P/R/F1, Macro F1 (primary selection metric),
    Weighted F1, and the confusion matrix, for one set of predictions."""
    labels = list(labels or classification_settings.labels)
    positive_label = positive_label or classification_settings.positive_label
    validate_predictions(df, tuple(labels))

    gold = df["gold_label"].to_numpy()
    pred = df["predicted_label"].to_numpy()

    accuracy = float(accuracy_score(gold, pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=labels, zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        gold, pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(gold, pred, labels=labels)

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    return {
        "n_examples": int(len(df)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "positive_label": positive_label,
        "sarcastic_precision": per_class[positive_label]["precision"],
        "sarcastic_recall": per_class[positive_label]["recall"],
        "sarcastic_f1": per_class[positive_label]["f1"],
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }
