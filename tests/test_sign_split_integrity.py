"""Regression tests against the *real* SIGN raw files, locking in the
counts verified by hand during the Phase 0 audit
(see SIGN_GENERALIZATION_PLAN.md section 1). If these numbers ever
change, either the raw files were edited (investigate before continuing,
per the brief) or `load_sign.py`'s parsing logic changed (update the plan
doc's audit numbers alongside these tests, don't just adjust the
assertion)."""
from __future__ import annotations

from src.sign.data.family_utils import assert_no_family_leakage
from src.sign.data.load_sign import load_family_table, summarize

EXPECTED = {
    "train": {
        "n_rows": 14292,
        "n_originals": 2292,
        "n_interpretations": 12000,
        "n_families": 2292,
        "n_clean_families": 2185,
        "n_anomalous_families": 107,
    },
    "dev": {
        "n_rows": 1770,
        "n_originals": 270,
        "n_interpretations": 1500,
        "n_families": 270,
        "n_clean_families": 240,
        "n_anomalous_families": 30,
    },
    "test": {
        "n_rows": 1735,
        "n_originals": 265,
        "n_interpretations": 1470,
        "n_families": 265,
        "n_clean_families": 237,
        "n_anomalous_families": 28,
    },
}


def test_real_sign_split_counts_match_the_audited_baseline():
    for split, expected in EXPECTED.items():
        df = load_family_table(split)
        assert summarize(df) == expected, f"split={split}"


def test_real_sign_splits_share_no_family_ids():
    # family_id is namespaced by split (e.g. "train-00001") so this should
    # always hold trivially -- guards against a future refactor that
    # drops the split prefix and silently reintroduces collisions.
    train = load_family_table("train")
    dev = load_family_table("dev")
    test = load_family_table("test")
    assert_no_family_leakage(
        train["family_id"], dev["family_id"], test["family_id"],
        labels=["train", "dev", "test"],
    )


def test_real_sign_every_family_has_exactly_one_original_row():
    for split in ("train", "dev", "test"):
        df = load_family_table(split)
        counts = df[df["role"] == "original"].groupby("family_id").size()
        assert (counts == 1).all(), f"split={split} has a family with != 1 original row"


def test_real_sign_example_ids_are_globally_unique_within_each_split():
    for split in ("train", "dev", "test"):
        df = load_family_table(split)
        assert df["example_id"].is_unique, f"split={split}"
