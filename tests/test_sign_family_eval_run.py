"""Tests for src/sign/family_eval/run_family_eval.py -- Phase 5's
Task A / Task B / Primary-Reference / per-rank blocks, computed over
hand-constructed prediction+family tables (no real model involved)."""
from __future__ import annotations

import math

import pandas as pd

from src.sign.family_eval.run_family_eval import (
    build_comparison_table,
    merge_with_family_table,
    per_interpretation_rank_recall,
    primary_reference_block,
    task_a_block,
    task_b_block,
)


def _family_table_row(family_id, role, interp_index, example_id, is_clean=True):
    return {
        "example_id": example_id,
        "family_id": family_id,
        "role": role,
        "interp_index": interp_index,
        "is_primary_interpretation": interp_index == 1,
        "is_clean_family": is_clean,
    }


def _one_family(family_id="f1", n_interps=5):
    family_rows = [_family_table_row(family_id, "original", 0, f"{family_id}-orig")]
    family_rows += [
        _family_table_row(family_id, "interpretation", i, f"{family_id}-interp{i}")
        for i in range(1, n_interps + 1)
    ]
    return pd.DataFrame(family_rows)


def _predictions(rows):
    """rows: list of (example_id, gold_label, predicted_label)."""
    return pd.DataFrame(rows, columns=["example_id", "gold_label", "predicted_label"])


def test_task_a_block_counts_only_originals():
    family_table = pd.concat([_one_family("f1"), _one_family("f2")], ignore_index=True)
    predictions = _predictions(
        [
            ("f1-orig", "sarcastic", "sarcastic"),  # detected
            ("f2-orig", "sarcastic", "not_sarcastic"),  # missed
        ]
        + [(f"f1-interp{i}", "not_sarcastic", "sarcastic") for i in range(1, 6)]
        + [(f"f2-interp{i}", "not_sarcastic", "sarcastic") for i in range(1, 6)]
    )
    merged = merge_with_family_table(predictions, family_table)
    block = task_a_block(merged)
    assert block["n_originals"] == 2
    assert block["n_correctly_predicted_sarcastic"] == 1
    assert block["sarcasm_detection_rate"] == 0.5
    assert block["false_negative_rate"] == 0.5


def test_task_b_block_uses_full_set_and_matches_compute_metrics_semantics():
    family_table = _one_family("f1")
    predictions = _predictions(
        [("f1-orig", "sarcastic", "sarcastic")]
        + [(f"f1-interp{i}", "not_sarcastic", "not_sarcastic") for i in range(1, 6)]
    )
    block = task_b_block(predictions)
    assert block["accuracy"] == 1.0
    assert block["macro_f1"] == 1.0


def test_primary_reference_block_only_uses_original_and_interp_1():
    family_table = _one_family("f1")
    predictions = _predictions(
        [("f1-orig", "sarcastic", "sarcastic")]
        + [("f1-interp1", "not_sarcastic", "not_sarcastic")]  # primary: correct
        + [(f"f1-interp{i}", "not_sarcastic", "sarcastic") for i in range(2, 6)]  # rest: wrong, ignored
    )
    merged = merge_with_family_table(predictions, family_table)
    block = primary_reference_block(merged)
    assert block["accuracy"] == 1.0  # only 2 rows considered, both correct
    assert block["pair_success_rate"] == 1.0
    assert block["n_pairs"] == 1


def test_primary_reference_pair_success_requires_both_original_and_interp1_correct():
    family_table = _one_family("f1")
    predictions = _predictions(
        [("f1-orig", "sarcastic", "sarcastic")]  # correct
        + [("f1-interp1", "not_sarcastic", "sarcastic")]  # wrong (primary should be not_sarcastic)
        + [(f"f1-interp{i}", "not_sarcastic", "not_sarcastic") for i in range(2, 6)]
    )
    merged = merge_with_family_table(predictions, family_table)
    block = primary_reference_block(merged)
    assert block["pair_success_rate"] == 0.0


def test_per_interpretation_rank_recall_is_computed_separately_per_rank():
    family_table = _one_family("f1")
    predictions = _predictions(
        [("f1-orig", "sarcastic", "sarcastic")]
        + [("f1-interp1", "not_sarcastic", "not_sarcastic")]  # rank 1: correct
        + [("f1-interp2", "not_sarcastic", "sarcastic")]  # rank 2: wrong
        + [(f"f1-interp{i}", "not_sarcastic", "not_sarcastic") for i in range(3, 6)]
    )
    merged = merge_with_family_table(predictions, family_table)
    ranks = per_interpretation_rank_recall(merged)
    assert ranks["rank_1"] == {"n": 1, "not_sarcastic_recall": 1.0}
    assert ranks["rank_2"] == {"n": 1, "not_sarcastic_recall": 0.0}
    assert ranks["rank_3"]["not_sarcastic_recall"] == 1.0


def test_per_interpretation_rank_recall_reports_zero_n_for_missing_rank():
    family_table = _one_family("f1", n_interps=3)  # anomalous family, no rank 4/5
    predictions = _predictions(
        [("f1-orig", "sarcastic", "sarcastic")]
        + [(f"f1-interp{i}", "not_sarcastic", "not_sarcastic") for i in range(1, 4)]
    )
    merged = merge_with_family_table(predictions, family_table)
    ranks = per_interpretation_rank_recall(merged)
    assert ranks["rank_4"]["n"] == 0
    assert math.isnan(ranks["rank_4"]["not_sarcastic_recall"])


def test_merge_with_family_table_raises_on_unmatched_example_id():
    family_table = _one_family("f1")
    predictions = _predictions([("does-not-exist", "sarcastic", "sarcastic")])
    try:
        merge_with_family_table(predictions, family_table)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_comparison_table_has_one_row_per_method_with_expected_columns():
    fake_result = {
        "method": "M_test",
        "experiment_id": "EXP-TEST",
        "task_a": {"sarcasm_detection_rate": 0.9, "false_negative_rate": 0.1},
        "task_b": {"macro_f1": 0.4, "accuracy": 0.45},
        "primary_reference": {"macro_f1": 0.6, "accuracy": 0.65, "pair_success_rate": 0.3},
        "per_interpretation_rank": {
            f"rank_{i}": {"n": 1, "not_sarcastic_recall": 0.5} for i in range(1, 6)
        },
        "view1_primary_reference_family": {
            "pairwise_contrastive_accuracy": 0.3,
            "strict_family_accuracy": 0.2,
        },
        "view2_full_family_all": {
            "pairwise_contrastive_accuracy": 0.2,
            "strict_family_accuracy": 0.05,
            "soft_family_score": {"mean_score": 0.3},
        },
        "view2_full_family_clean_only": {
            "strict_family_accuracy": 0.06,
            "soft_family_score": {"mean_score": 0.31},
        },
    }
    table = build_comparison_table([fake_result, fake_result])
    assert len(table) == 2
    assert "task_a_detection_rate" in table.columns
    assert "primary_ref_pair_success_rate" in table.columns
    assert "view2_all_soft_family_score" in table.columns
