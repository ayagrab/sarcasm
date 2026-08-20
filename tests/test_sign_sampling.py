"""Tests for src/sign/data/family_utils.py -- deterministic family-level
sampling (Phase 7's foundation: balanced/fractional/interpretation-count
variants must all be reproducible from a recorded seed)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.sign.data.family_utils import (
    sample_family_ids,
    select_families,
    select_k_interpretations_per_family,
    select_primary_interpretation_per_family,
    select_top_k_interpretations_per_family,
)


def _family_df(n_families=4, interps_per_family=5):
    rows = []
    for f in range(n_families):
        fam = f"train-{f:05d}"
        rows.append(
            {"family_id": fam, "role": "original", "interp_index": 0, "text": f"orig{f}", "label": "sarcastic", "example_id": f"{fam}-orig", "family_size": interps_per_family, "is_clean_family": interps_per_family == 5}
        )
        for i in range(1, interps_per_family + 1):
            rows.append(
                {"family_id": fam, "role": "interpretation", "interp_index": i, "text": f"i{f}-{i}", "label": "not_sarcastic", "example_id": f"{fam}-interp{i}", "family_size": interps_per_family, "is_clean_family": interps_per_family == 5}
            )
    return pd.DataFrame(rows)


def test_sample_family_ids_is_deterministic_given_a_seed():
    ids = [f"train-{i:05d}" for i in range(100)]
    a = sample_family_ids(ids, seed=42, frac=0.25)
    b = sample_family_ids(ids, seed=42, frac=0.25)
    assert a == b


def test_sample_family_ids_different_seeds_usually_differ():
    ids = [f"train-{i:05d}" for i in range(100)]
    a = sample_family_ids(ids, seed=1, frac=0.5)
    b = sample_family_ids(ids, seed=2, frac=0.5)
    assert a != b


def test_sample_family_ids_respects_frac():
    ids = [f"train-{i:05d}" for i in range(100)]
    sampled = sample_family_ids(ids, seed=42, frac=0.1)
    assert len(sampled) == 10


def test_sample_family_ids_respects_n():
    ids = [f"train-{i:05d}" for i in range(100)]
    sampled = sample_family_ids(ids, seed=42, n=7)
    assert len(sampled) == 7


def test_sample_family_ids_rejects_both_frac_and_n():
    with pytest.raises(ValueError):
        sample_family_ids(["a", "b"], seed=42, frac=0.5, n=1)


def test_sample_family_ids_order_independent_of_input_order():
    ids_a = [f"train-{i:05d}" for i in range(50)]
    ids_b = list(reversed(ids_a))
    assert sample_family_ids(ids_a, seed=42, frac=0.3) == sample_family_ids(ids_b, seed=42, frac=0.3)


def test_select_families_returns_whole_families_only():
    df = _family_df(n_families=4)
    subset = select_families(df, ["train-00001", "train-00003"])
    assert set(subset["family_id"]) == {"train-00001", "train-00003"}
    # every selected family keeps its original + all 5 interpretations
    assert len(subset) == 2 * 6


def test_select_k_interpretations_per_family_keeps_the_original(tmp_path=None):
    df = _family_df(n_families=2, interps_per_family=5)
    reduced = select_k_interpretations_per_family(df, k=2, seed=42)
    for fam, group in reduced.groupby("family_id"):
        assert (group["role"] == "original").sum() == 1
        assert (group["role"] == "interpretation").sum() == 2


def test_select_k_interpretations_per_family_is_deterministic():
    df = _family_df(n_families=3, interps_per_family=5)
    a = select_k_interpretations_per_family(df, k=2, seed=42)
    b = select_k_interpretations_per_family(df, k=2, seed=42)
    pd.testing.assert_frame_equal(
        a.sort_values("example_id").reset_index(drop=True),
        b.sort_values("example_id").reset_index(drop=True),
    )


def test_select_k_interpretations_per_family_selections_are_nested():
    """k=1's interpretation must be contained in k=2's, which must be
    contained in k=3's -- this is what makes Phase 10's ablation a
    controlled comparison (adding interpretations, not swapping them)."""
    df = _family_df(n_families=5, interps_per_family=5)
    picks = {}
    for k in (1, 2, 3, 5):
        reduced = select_k_interpretations_per_family(df, k=k, seed=7)
        interp_ids = set(reduced[reduced["role"] == "interpretation"]["example_id"])
        picks[k] = interp_ids

    assert picks[1] <= picks[2] <= picks[3] <= picks[5]
    assert len(picks[5]) == 5 * 5  # every interpretation, for 5 families


def test_select_top_k_interpretations_is_rank_based_not_shuffled():
    """k=1 must always be interpretation #1 specifically (interp_index==1),
    never a random pick -- this is the whole point of the rank-based
    selector (2026-08-20 "interpretation #1 is primary" clarification)."""
    df = _family_df(n_families=5, interps_per_family=5)
    top1 = select_top_k_interpretations_per_family(df, k=1)
    interps = top1[top1["role"] == "interpretation"]
    assert (interps["interp_index"] == 1).all()
    assert len(interps) == 5  # exactly one interpretation per family


def test_select_top_k_interpretations_is_deterministic_with_no_seed():
    df = _family_df(n_families=4, interps_per_family=5)
    a = select_top_k_interpretations_per_family(df, k=3)
    b = select_top_k_interpretations_per_family(df, k=3)
    pd.testing.assert_frame_equal(
        a.sort_values("example_id").reset_index(drop=True),
        b.sort_values("example_id").reset_index(drop=True),
    )


def test_select_top_k_interpretations_selections_are_nested_by_rank():
    df = _family_df(n_families=5, interps_per_family=5)
    picks = {}
    for k in (1, 2, 3, 5):
        reduced = select_top_k_interpretations_per_family(df, k=k)
        picks[k] = set(reduced[reduced["role"] == "interpretation"]["interp_index"])
    assert picks[1] == {1}
    assert picks[2] == {1, 2}
    assert picks[3] == {1, 2, 3}
    assert picks[5] == {1, 2, 3, 4, 5}


def test_select_top_k_interpretations_does_not_invent_missing_interpretations():
    """A family with only 3 interpretations available must contribute at
    most 3, even when k=5 -- never padded/invented."""
    df = _family_df(n_families=1, interps_per_family=3)
    reduced = select_top_k_interpretations_per_family(df, k=5)
    interps = reduced[reduced["role"] == "interpretation"]
    assert len(interps) == 3


def test_select_top_k_interpretations_rejects_k_below_one():
    df = _family_df(n_families=1, interps_per_family=5)
    with pytest.raises(ValueError):
        select_top_k_interpretations_per_family(df, k=0)


def test_select_primary_interpretation_per_family_matches_top_k_one():
    df = _family_df(n_families=3, interps_per_family=5)
    primary = select_primary_interpretation_per_family(df)
    top1 = select_top_k_interpretations_per_family(df, k=1)
    pd.testing.assert_frame_equal(
        primary.sort_values("example_id").reset_index(drop=True),
        top1.sort_values("example_id").reset_index(drop=True),
    )
    interps = primary[primary["role"] == "interpretation"]
    assert len(interps) == 3  # one primary interpretation per family
    assert (interps["interp_index"] == 1).all()
