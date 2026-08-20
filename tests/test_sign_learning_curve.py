"""Tests for src/sign/learning_curve/run_m1_learning_curve.py's
`build_sign_fraction` -- Phase 9's family-level fractional sampling,
shared by both the M1 and M6 legs."""
from __future__ import annotations

import pandas as pd

from src.sign.learning_curve.run_m1_learning_curve import build_sign_fraction


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


def _train_table(n_families=20, n_interps=5):
    rows = []
    for f in range(n_families):
        fam = f"train-{f:05d}"
        rows.append(_family_row(fam, "original", 0, f"orig{f}", "sarcastic", n_interps))
        for i in range(1, n_interps + 1):
            rows.append(_family_row(fam, "interpretation", i, f"interp{f}-{i}", "not_sarcastic", n_interps))
    return pd.DataFrame(rows)


def test_build_sign_fraction_respects_family_level_frac():
    train = _train_table(n_families=20)
    fraction = build_sign_fraction(train, frac=0.25, seed=42)
    assert fraction["family_id"].nunique() == 5  # 25% of 20 families


def test_build_sign_fraction_is_balanced_one_to_one():
    train = _train_table(n_families=10)
    fraction = build_sign_fraction(train, frac=0.5, seed=42)
    assert (fraction["role"] == "original").sum() == (fraction["role"] == "interpretation").sum()


def test_build_sign_fraction_uses_rank_1_never_random():
    train = _train_table(n_families=10)
    fraction = build_sign_fraction(train, frac=0.5, seed=42)
    interps = fraction[fraction["role"] == "interpretation"]
    assert (interps["interp_index"] == 1).all()


def test_build_sign_fraction_is_deterministic_given_seed():
    train = _train_table(n_families=10)
    a = build_sign_fraction(train, frac=0.3, seed=7)
    b = build_sign_fraction(train, frac=0.3, seed=7)
    pd.testing.assert_frame_equal(
        a.sort_values("example_id").reset_index(drop=True),
        b.sort_values("example_id").reset_index(drop=True),
    )
