"""Tests for src/classification/llm/few_shot_selection.py. Selection must
only ever draw from the given (TRAIN-only, by construction of the caller)
dataframe, and must be deterministic given a seed."""
from __future__ import annotations

import pandas as pd

from src.classification.llm.few_shot_selection import select_curated_few_shot, select_random_few_shot


def _fixture_train_df():
    rows = []
    for category in ("GEN", "HYP", "RQ"):
        for label in ("sarcastic", "not_sarcastic"):
            for i in range(10):
                rows.append(
                    {
                        "example_id": f"{category}-{label}-{i}",
                        "category": category,
                        "label": label,
                        "text": f"{category} {label} example {i}",
                    }
                )
    return pd.DataFrame(rows)


def test_select_random_few_shot_returns_requested_count():
    demo = select_random_few_shot(_fixture_train_df(), n_shots=8, seed=42)
    assert len(demo) == 8


def test_select_random_few_shot_balances_labels():
    demo = select_random_few_shot(_fixture_train_df(), n_shots=8, seed=42)
    counts = demo["label"].value_counts()
    assert counts["sarcastic"] == 4
    assert counts["not_sarcastic"] == 4


def test_select_random_few_shot_is_deterministic():
    a = select_random_few_shot(_fixture_train_df(), n_shots=8, seed=42)
    b = select_random_few_shot(_fixture_train_df(), n_shots=8, seed=42)
    assert sorted(a["example_id"]) == sorted(b["example_id"])


def test_select_random_few_shot_different_seeds_can_differ():
    a = select_random_few_shot(_fixture_train_df(), n_shots=8, seed=1)
    b = select_random_few_shot(_fixture_train_df(), n_shots=8, seed=2)
    assert sorted(a["example_id"]) != sorted(b["example_id"])


def test_select_curated_few_shot_covers_multiple_categories():
    demo = select_curated_few_shot(_fixture_train_df(), n_shots=6, seed=42)
    assert len(demo) == 6
    assert demo["category"].nunique() > 1


def test_select_curated_few_shot_is_deterministic():
    a = select_curated_few_shot(_fixture_train_df(), n_shots=6, seed=42)
    b = select_curated_few_shot(_fixture_train_df(), n_shots=6, seed=42)
    assert sorted(a["example_id"]) == sorted(b["example_id"])
