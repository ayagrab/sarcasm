"""Tests for src/sign/train_prep/build_train_variants.py -- Phase 7's
data-prep-only SIGN Train variant construction (no training happens
here). Hand-constructed family tables so leakage/balance/nesting
guarantees are pinned down before touching the real 12,000-row Train
file."""
from __future__ import annotations

import pandas as pd
import pytest

from src.sign.data.family_utils import assert_no_family_leakage
from src.sign.train_prep.build_train_variants import (
    VARIANT_COLUMNS,
    build_k_variant,
    build_primary_variant,
    variant_metadata,
    verify_no_cross_split_leakage,
)


def _family_row(family_id, role, interp_index, text, label, family_size=5, is_clean=True):
    return {
        "family_id": family_id,
        "split": "train",
        "role": role,
        "interp_index": interp_index,
        "is_primary_interpretation": interp_index == 1,
        "text": text,
        "label": label,
        "example_id": f"{family_id}-{role}{interp_index}",
        "family_size": family_size,
        "is_clean_family": is_clean,
    }


def _train_table(n_families=3, n_interps=5):
    rows = []
    for f in range(n_families):
        fam = f"train-{f:05d}"
        rows.append(_family_row(fam, "original", 0, f"orig{f}", "sarcastic", n_interps))
        for i in range(1, n_interps + 1):
            rows.append(_family_row(fam, "interpretation", i, f"interp{f}-{i}", "not_sarcastic", n_interps))
    return pd.DataFrame(rows)


def test_primary_variant_is_balanced_one_to_one():
    train = _train_table(n_families=4)
    variant = build_primary_variant(train)
    assert (variant["role"] == "original").sum() == (variant["role"] == "interpretation").sum() == 4
    assert set(variant.columns) == set(VARIANT_COLUMNS)


def test_primary_variant_uses_rank_1_never_random():
    train = _train_table(n_families=2)
    variant = build_primary_variant(train)
    interps = variant[variant["role"] == "interpretation"]
    assert (interps["interp_index"] == 1).all()


def test_k_variants_are_nested_by_rank():
    train = _train_table(n_families=3)
    picks = {}
    for k in (2, 3, 5):
        v = build_k_variant(train, k)
        picks[k] = set(v.loc[v["role"] == "interpretation", "interp_index"])
    assert picks[2] == {1, 2}
    assert picks[3] == {1, 2, 3}
    assert picks[5] == {1, 2, 3, 4, 5}


def test_k_variant_is_imbalanced_by_construction():
    train = _train_table(n_families=3)
    v = build_k_variant(train, 3)
    n_orig = (v["role"] == "original").sum()
    n_interp = (v["role"] == "interpretation").sum()
    assert n_interp == 3 * n_orig  # 3 interpretations per 1 original -> imbalanced


def test_verify_no_cross_split_leakage_passes_for_disjoint_families():
    train = _train_table(n_families=2)
    dev = _train_table(n_families=1)
    dev["family_id"] = "dev-00000"  # distinct from train's family_ids
    test = _train_table(n_families=1)
    test["family_id"] = "test-00000"
    variant = build_primary_variant(train)
    verify_no_cross_split_leakage(variant, dev, test)  # should not raise


def test_verify_no_cross_split_leakage_raises_on_overlap():
    train = _train_table(n_families=2)
    variant = build_primary_variant(train)
    dev_with_overlap = train.copy()  # shares family_ids with train -> leakage
    test = _train_table(n_families=1)
    test["family_id"] = "test-00000"
    with pytest.raises(ValueError):
        verify_no_cross_split_leakage(variant, dev_with_overlap, test)


def test_variant_metadata_reports_expected_counts():
    train = _train_table(n_families=5)
    variant = build_primary_variant(train)
    meta = variant_metadata("primary", variant, None)
    assert meta["n_families"] == 5
    assert meta["n_originals"] == 5
    assert meta["n_interpretations"] == 5
    assert meta["interp_ranks_included"] == [1]
    assert meta["class_balance"] == {"sarcastic": 5, "not_sarcastic": 5}
    assert "duplicate_interpretation1_limitation" in meta


def test_k_variant_does_not_invent_missing_interpretations_for_anomalous_family():
    train = _train_table(n_families=1, n_interps=3)  # anomalous: only 3 interps available
    v = build_k_variant(train, 5)
    assert (v["role"] == "interpretation").sum() == 3
