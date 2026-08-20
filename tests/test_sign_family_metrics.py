"""Tests for src/sign/family_eval/metrics.py -- family-aware SIGN metrics
(Phase 5). Pure computation over hand-constructed prediction tables so
every metric's exact definition is pinned down before any real model
predictions exist."""
from __future__ import annotations

import math

import pandas as pd

from src.sign.family_eval.metrics import (
    compute_family_metrics,
    interpretation_non_sarcasm_rate,
    original_sarcasm_detection_rate,
    pairwise_contrastive_accuracy,
    soft_family_score,
    strict_family_accuracy,
)


def _row(family_id, role, predicted_label):
    return {"family_id": family_id, "role": role, "predicted_label": predicted_label}


def test_perfect_predictions_score_perfectly_on_every_metric():
    rows = [_row("f1", "original", "sarcastic")]
    rows += [_row("f1", "interpretation", "not_sarcastic") for _ in range(5)]
    df = pd.DataFrame(rows)

    assert original_sarcasm_detection_rate(df) == 1.0
    assert interpretation_non_sarcasm_rate(df) == 1.0
    assert pairwise_contrastive_accuracy(df) == 1.0
    assert strict_family_accuracy(df) == 1.0
    soft = soft_family_score(df)
    assert soft["mean_score"] == 1.0
    assert soft["n_families"] == 1


def test_missed_original_zeroes_out_pairwise_and_strict_but_not_soft_denominator():
    rows = [_row("f1", "original", "not_sarcastic")]  # missed!
    rows += [_row("f1", "interpretation", "not_sarcastic") for _ in range(5)]  # all correct
    df = pd.DataFrame(rows)

    assert original_sarcasm_detection_rate(df) == 0.0
    assert interpretation_non_sarcasm_rate(df) == 1.0
    assert pairwise_contrastive_accuracy(df) == 0.0  # original wrong -> every pair fails
    assert strict_family_accuracy(df) == 0.0  # one wrong row in the family
    soft = soft_family_score(df)
    assert math.isnan(soft["mean_score"])  # no family had a correct original
    assert soft["n_families"] == 0


def test_false_positive_interpretation_lowers_interp_rate_and_pairwise_but_original_still_detected():
    rows = [_row("f1", "original", "sarcastic")]
    rows += [_row("f1", "interpretation", "not_sarcastic") for _ in range(4)]
    rows += [_row("f1", "interpretation", "sarcastic")]  # 1 false positive
    df = pd.DataFrame(rows)

    assert original_sarcasm_detection_rate(df) == 1.0
    assert interpretation_non_sarcasm_rate(df) == 4 / 5
    assert pairwise_contrastive_accuracy(df) == 4 / 5
    assert strict_family_accuracy(df) == 0.0  # not ALL interpretations correct
    soft = soft_family_score(df)
    assert soft["mean_score"] == 4 / 5
    assert soft["n_families"] == 1


def test_metrics_aggregate_correctly_across_multiple_families():
    rows = []
    # f1: everything correct
    rows.append(_row("f1", "original", "sarcastic"))
    rows += [_row("f1", "interpretation", "not_sarcastic") for _ in range(5)]
    # f2: original missed
    rows.append(_row("f2", "original", "not_sarcastic"))
    rows += [_row("f2", "interpretation", "not_sarcastic") for _ in range(5)]
    df = pd.DataFrame(rows)

    assert original_sarcasm_detection_rate(df) == 0.5  # 1 of 2 originals correct
    assert interpretation_non_sarcasm_rate(df) == 1.0  # all 10 interpretations correct
    assert pairwise_contrastive_accuracy(df) == 0.5  # 5 of 10 pairs succeed (f1's)
    assert strict_family_accuracy(df) == 0.5  # only f1 fully correct
    soft = soft_family_score(df)
    assert soft["mean_score"] == 1.0  # only f1 counted (correct original), and it's perfect
    assert soft["n_families"] == 1


def test_compute_family_metrics_returns_all_keys():
    rows = [_row("f1", "original", "sarcastic")]
    rows += [_row("f1", "interpretation", "not_sarcastic") for _ in range(5)]
    df = pd.DataFrame(rows)
    metrics = compute_family_metrics(df)
    assert set(metrics.keys()) == {
        "original_sarcasm_detection_rate",
        "interpretation_non_sarcasm_rate",
        "pairwise_contrastive_accuracy",
        "strict_family_accuracy",
        "soft_family_score",
    }
