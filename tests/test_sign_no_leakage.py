"""Tests for src/sign/data/family_utils.py -- the leakage guard itself,
and that it actually catches the leakage patterns the brief warns about
(interpretations of one family split across a train/eval boundary)."""
from __future__ import annotations

import pytest

from src.sign.data.family_utils import assert_no_family_leakage


def test_assert_no_family_leakage_passes_for_disjoint_groups():
    assert_no_family_leakage(["a", "b"], ["c", "d"], ["e"])  # no exception


def test_assert_no_family_leakage_raises_on_overlap():
    with pytest.raises(ValueError):
        assert_no_family_leakage(["a", "b"], ["b", "c"])


def test_assert_no_family_leakage_reports_which_groups_and_which_ids():
    with pytest.raises(ValueError) as exc_info:
        assert_no_family_leakage(["fam-1", "fam-2"], ["fam-2"], labels=["train", "test"])
    msg = str(exc_info.value)
    assert "train" in msg and "test" in msg
    assert "fam-2" in msg


def test_assert_no_family_leakage_catches_a_family_split_across_three_groups():
    # simulates the exact scenario the brief warns about: one family's
    # interpretations scattered across train/dev/test.
    train_ids = ["fam-1-interp1", "fam-1-interp2"]
    dev_ids = ["fam-1-interp3"]
    test_ids = ["fam-1-interp4", "fam-1-interp5"]
    # here the leakage is at the *interpretation* id level belonging to
    # one family -- assert_no_family_leakage operates on whatever ID
    # column it's given, so callers must pass family_id (not example_id)
    # for this to be a meaningful family-level check.
    assert_no_family_leakage(train_ids, dev_ids, test_ids)  # no exception: distinct example_ids

    family_ids_per_split = [["fam-1"], ["fam-1"], ["fam-1"]]
    with pytest.raises(ValueError):
        assert_no_family_leakage(*family_ids_per_split)


def test_assert_no_family_leakage_accepts_dataframe_columns():
    import pandas as pd

    train = pd.DataFrame({"family_id": ["a", "b"]})
    test = pd.DataFrame({"family_id": ["c", "d"]})
    assert_no_family_leakage(train["family_id"], test["family_id"])  # no exception

    test_bad = pd.DataFrame({"family_id": ["a", "d"]})
    with pytest.raises(ValueError):
        assert_no_family_leakage(train["family_id"], test_bad["family_id"])
