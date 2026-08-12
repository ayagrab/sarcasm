"""Tests for src/classification/evaluation/error_analysis.py -- pure
pandas logic, no GPU/model needed. Uses tmp_path fixture result dirs
instead of real experiment output."""
from __future__ import annotations

import pandas as pd
import pytest

from src.classification.evaluation.error_analysis import (
    build_disagreement_table,
    full_agreement_rate,
    load_predictions,
    pairwise_breakdown,
    summarize_pairwise,
)


def _write_predictions(results_dir, experiment_id, rows):
    out_dir = results_dir / experiment_id
    out_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)


@pytest.fixture
def fixture_results_dir(tmp_path):
    _write_predictions(
        tmp_path,
        "EXP-A",
        [
            {"example_id": "1", "gold_label": "sarcastic", "predicted_label": "sarcastic"},
            {"example_id": "2", "gold_label": "not_sarcastic", "predicted_label": "sarcastic"},
            {"example_id": "3", "gold_label": "sarcastic", "predicted_label": "not_sarcastic"},
            {"example_id": "4", "gold_label": "not_sarcastic", "predicted_label": "not_sarcastic"},
        ],
    )
    _write_predictions(
        tmp_path,
        "EXP-B",
        [
            {"example_id": "1", "gold_label": "sarcastic", "predicted_label": "sarcastic"},
            {"example_id": "2", "gold_label": "not_sarcastic", "predicted_label": "not_sarcastic"},
            {"example_id": "3", "gold_label": "sarcastic", "predicted_label": "sarcastic"},
            {"example_id": "4", "gold_label": "not_sarcastic", "predicted_label": "sarcastic"},
        ],
    )
    return tmp_path


def test_load_predictions_renames_column(fixture_results_dir):
    df = load_predictions("EXP-A", "model_a", results_dir=fixture_results_dir)
    assert "pred__model_a" in df.columns
    assert len(df) == 4


def test_load_predictions_missing_experiment_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_predictions("NOPE", "model_a", results_dir=tmp_path)


def test_build_disagreement_table_merges_on_example_id(fixture_results_dir):
    table = build_disagreement_table(
        {"model_a": "EXP-A", "model_b": "EXP-B"}, results_dir=fixture_results_dir
    )
    assert len(table) == 4
    assert {"example_id", "gold_label", "pred__model_a", "pred__model_b"} <= set(table.columns)


def test_pairwise_breakdown_categories(fixture_results_dir):
    table = build_disagreement_table(
        {"model_a": "EXP-A", "model_b": "EXP-B"}, results_dir=fixture_results_dir
    )
    breakdown = pairwise_breakdown(table, "model_a", "model_b")
    # example 1: A correct, B correct -> both_right
    # example 2: A wrong (sarcastic), B correct -> model_b right, model_a wrong
    # example 3: A wrong (not_sarcastic), B correct -> model_b right, model_a wrong
    # example 4: A correct, B wrong (sarcastic) -> model_a right, model_b wrong
    assert set(breakdown["both_right"]["example_id"]) == {1}
    assert set(breakdown["both_wrong"]["example_id"]) == set()
    assert set(breakdown["model_b_right_model_a_wrong"]["example_id"]) == {2, 3}
    assert set(breakdown["model_a_right_model_b_wrong"]["example_id"]) == {4}


def test_pairwise_breakdown_unknown_model_raises(fixture_results_dir):
    table = build_disagreement_table(
        {"model_a": "EXP-A", "model_b": "EXP-B"}, results_dir=fixture_results_dir
    )
    with pytest.raises(KeyError):
        pairwise_breakdown(table, "model_a", "nonexistent")


def test_summarize_pairwise_counts(fixture_results_dir):
    table = build_disagreement_table(
        {"model_a": "EXP-A", "model_b": "EXP-B"}, results_dir=fixture_results_dir
    )
    summary = summarize_pairwise(table, "model_a", "model_b")
    assert summary == {
        "model_a_right_model_b_wrong": 1,
        "model_b_right_model_a_wrong": 2,
        "both_wrong": 0,
        "both_right": 1,
    }


def test_full_agreement_rate(fixture_results_dir):
    table = build_disagreement_table(
        {"model_a": "EXP-A", "model_b": "EXP-B"}, results_dir=fixture_results_dir
    )
    # only example_id=1 has identical predictions across both models
    assert full_agreement_rate(table, ["model_a", "model_b"]) == pytest.approx(0.25)
