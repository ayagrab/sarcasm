"""Tests for src/sign/interp_ablation/*.py -- Phase 10's interpretation-
count ablation imbalance-handling helpers."""
from __future__ import annotations

import pandas as pd

from src.sign.interp_ablation.run_m1_interp_ablation import build_balanced_pipeline
from src.sign.interp_ablation.run_m6_interp_ablation import duplicate_originals_for_balance


def test_build_balanced_pipeline_sets_class_weight_balanced():
    pipeline = build_balanced_pipeline(seed=42)
    assert pipeline.named_steps["clf"].class_weight == "balanced"


def _k_variant(n_families=3, k=3):
    rows = []
    for f in range(n_families):
        fam = f"train-{f:05d}"
        rows.append({"family_id": fam, "role": "original", "example_id": f"{fam}-orig", "text": f"o{f}", "label": "sarcastic"})
        for i in range(1, k + 1):
            rows.append({"family_id": fam, "role": "interpretation", "example_id": f"{fam}-i{i}", "text": f"i{f}-{i}", "label": "not_sarcastic"})
    return pd.DataFrame(rows)


def test_duplicate_originals_for_balance_restores_one_to_one():
    df = _k_variant(n_families=4, k=3)
    balanced = duplicate_originals_for_balance(df, k=3)
    assert (balanced["role"] == "original").sum() == (balanced["role"] == "interpretation").sum()


def test_duplicate_originals_for_balance_keeps_example_ids_unique():
    df = _k_variant(n_families=2, k=2)
    balanced = duplicate_originals_for_balance(df, k=2)
    assert balanced["example_id"].duplicated().sum() == 0


def test_duplicate_originals_for_balance_preserves_interpretation_rows_unchanged():
    df = _k_variant(n_families=2, k=2)
    balanced = duplicate_originals_for_balance(df, k=2)
    original_interps = df[df["role"] == "interpretation"]
    balanced_interps = balanced[balanced["role"] == "interpretation"]
    assert len(original_interps) == len(balanced_interps)
    assert set(original_interps["example_id"]) == set(balanced_interps["example_id"])
