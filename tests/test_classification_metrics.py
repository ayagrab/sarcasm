"""Tests for src/classification/evaluation/metrics.py -- the single shared
metrics implementation used by every approach (M1-M6)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.classification.evaluation.metrics import compute_metrics, validate_predictions


def _predictions_df():
    return pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d"],
            "gold_label": ["sarcastic", "sarcastic", "not_sarcastic", "not_sarcastic"],
            "predicted_label": ["sarcastic", "not_sarcastic", "not_sarcastic", "not_sarcastic"],
        }
    )


def test_compute_metrics_accuracy():
    metrics = compute_metrics(_predictions_df())
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["n_examples"] == 4


def test_compute_metrics_macro_f1_matches_manual_calc():
    # sarcastic: TP=1, FN=1, FP=0 -> P=1.0, R=0.5, F1=0.6667
    # not_sarcastic: TP=2, FN=0, FP=1 -> P=0.6667, R=1.0, F1=0.8
    metrics = compute_metrics(_predictions_df())
    assert metrics["per_class"]["sarcastic"]["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["per_class"]["not_sarcastic"]["f1"] == pytest.approx(0.8, abs=1e-4)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2, abs=1e-4)


def test_compute_metrics_confusion_matrix_shape_and_order():
    metrics = compute_metrics(_predictions_df())
    cm = metrics["confusion_matrix"]
    assert cm["labels"] == ["not_sarcastic", "sarcastic"]
    assert len(cm["matrix"]) == 2 and len(cm["matrix"][0]) == 2


def test_compute_metrics_sarcastic_shortcuts_match_per_class():
    metrics = compute_metrics(_predictions_df())
    assert metrics["sarcastic_f1"] == metrics["per_class"]["sarcastic"]["f1"]
    assert metrics["sarcastic_precision"] == metrics["per_class"]["sarcastic"]["precision"]
    assert metrics["sarcastic_recall"] == metrics["per_class"]["sarcastic"]["recall"]


def test_validate_predictions_raises_on_duplicate_example_id():
    df = _predictions_df()
    df.loc[4] = ["a", "sarcastic", "sarcastic"]
    with pytest.raises(ValueError, match="Duplicate example_id"):
        validate_predictions(df)


def test_validate_predictions_raises_on_missing_column():
    df = _predictions_df().drop(columns=["gold_label"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_predictions(df)


def test_validate_predictions_raises_on_invalid_predicted_label():
    df = _predictions_df()
    df.loc[0, "predicted_label"] = "SARCASTIC!!"
    with pytest.raises(ValueError, match="not normalized"):
        validate_predictions(df)


def test_validate_predictions_raises_on_invalid_gold_label():
    df = _predictions_df()
    df.loc[0, "gold_label"] = "unknown"
    with pytest.raises(ValueError, match="Unexpected gold_label"):
        validate_predictions(df)
