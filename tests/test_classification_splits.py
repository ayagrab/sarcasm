"""Tests for src/classification/data/make_splits.py -- the grouped,
stratified canonical split. The key property under test is that no
duplicate-text group (`dup_group_id`) ever spans more than one split."""
from __future__ import annotations

import pandas as pd
import pytest

from src.classification.data.make_splits import _assert_no_group_leakage, make_splits


@pytest.fixture
def fixture_canonical_df():
    # 40 singleton-group rows (20 sarcastic, 20 not) + 10 rows across 5
    # duplicate-text groups (2 rows per group) -- enough for a 60/20/20 split
    # with StratifiedGroupKFold to have room to work with.
    rows = []
    for i in range(20):
        rows.append({"example_id": f"S-{i}", "dup_group_id": 1000 + i, "category": "GEN", "label": "sarcastic", "text": f"s{i}"})
        rows.append({"example_id": f"N-{i}", "dup_group_id": 2000 + i, "category": "GEN", "label": "not_sarcastic", "text": f"n{i}"})
    for i in range(5):
        group_id = 3000 + i
        rows.append({"example_id": f"D-{i}-a", "dup_group_id": group_id, "category": "GEN", "label": "sarcastic", "text": f"dup{i}"})
        rows.append({"example_id": f"D-{i}-b", "dup_group_id": group_id, "category": "RQ", "label": "sarcastic", "text": f"dup{i}"})
    return pd.DataFrame(rows)


def test_make_splits_never_splits_a_duplicate_group_across_partitions(fixture_canonical_df):
    assignments = make_splits(fixture_canonical_df, 0.6, 0.2, 0.2, seed=42)
    group_splits = assignments.groupby("dup_group_id")["split"].nunique()
    assert (group_splits == 1).all()


def test_make_splits_covers_every_row_exactly_once(fixture_canonical_df):
    assignments = make_splits(fixture_canonical_df, 0.6, 0.2, 0.2, seed=42)
    assert set(assignments["example_id"]) == set(fixture_canonical_df["example_id"])
    assert assignments["split"].isin(["train", "dev", "test"]).all()


def test_make_splits_roughly_matches_requested_fractions(fixture_canonical_df):
    assignments = make_splits(fixture_canonical_df, 0.6, 0.2, 0.2, seed=42)
    fracs = assignments["split"].value_counts(normalize=True)
    assert fracs["train"] == pytest.approx(0.6, abs=0.15)
    assert fracs["test"] == pytest.approx(0.2, abs=0.15)


def test_make_splits_rejects_fractions_not_summing_to_one(fixture_canonical_df):
    with pytest.raises(ValueError):
        make_splits(fixture_canonical_df, 0.5, 0.3, 0.3, seed=42)


def test_make_splits_is_deterministic_given_a_seed(fixture_canonical_df):
    a = make_splits(fixture_canonical_df, 0.6, 0.2, 0.2, seed=7)
    b = make_splits(fixture_canonical_df, 0.6, 0.2, 0.2, seed=7)
    assert (a.sort_values("example_id")["split"].values == b.sort_values("example_id")["split"].values).all()


def test_assert_no_group_leakage_raises_when_violated():
    bad = pd.DataFrame(
        {
            "example_id": ["a", "b"],
            "dup_group_id": [1, 1],
            "split": ["train", "test"],
        }
    )
    with pytest.raises(AssertionError):
        _assert_no_group_leakage(bad)
