"""Phase 8's mandatory post-adaptation error-analysis repeat (SIGN_
GENERALIZATION_PLAN.md, Phase 8): an explicit before (zero-transfer) vs.
after (adapted) diff for one model's SIGN Test predictions -- fixed /
still-wrong / newly-broken, for both originals (false negatives) and
interpretations (false positives). A macro-F1 delta alone doesn't answer
"which failures did adaptation actually fix," this does.
"""
from __future__ import annotations

import pandas as pd


def originals_correctness(predictions: pd.DataFrame, family_table: pd.DataFrame) -> pd.DataFrame:
    """`predictions`: example_id, gold_label, predicted_label. Returns
    one row per original: family_id, correct (predicted sarcastic)."""
    merged = predictions.merge(family_table[["example_id", "family_id", "role"]], on="example_id", how="left")
    originals = merged[merged["role"] == "original"].copy()
    originals["correct"] = originals["predicted_label"] == "sarcastic"
    return originals[["family_id", "correct"]]


def interpretations_correctness(predictions: pd.DataFrame, family_table: pd.DataFrame) -> pd.DataFrame:
    """One row per interpretation: family_id, interp_index, correct
    (predicted not_sarcastic)."""
    merged = predictions.merge(
        family_table[["example_id", "family_id", "role", "interp_index"]], on="example_id", how="left"
    )
    interps = merged[merged["role"] == "interpretation"].copy()
    interps["correct"] = interps["predicted_label"] == "not_sarcastic"
    return interps[["family_id", "interp_index", "correct"]]


def _diff(merged: pd.DataFrame) -> dict:
    fixed = merged[(~merged["correct_before"]) & (merged["correct_after"])]
    still_wrong = merged[(~merged["correct_before"]) & (~merged["correct_after"])]
    newly_broken = merged[(merged["correct_before"]) & (~merged["correct_after"])]
    still_correct = merged[(merged["correct_before"]) & (merged["correct_after"])]
    return {
        "n_total": int(len(merged)),
        "n_fixed": int(len(fixed)),
        "n_still_wrong": int(len(still_wrong)),
        "n_newly_broken": int(len(newly_broken)),
        "n_still_correct": int(len(still_correct)),
    }


def diff_originals(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    merged = before.merge(after, on="family_id", suffixes=("_before", "_after"))
    result = _diff(merged)
    fixed = merged[(~merged["correct_before"]) & (merged["correct_after"])]
    still_wrong = merged[(~merged["correct_before"]) & (~merged["correct_after"])]
    newly_broken = merged[(merged["correct_before"]) & (~merged["correct_after"])]
    result["fixed_family_ids"] = sorted(fixed["family_id"].tolist())
    result["still_wrong_family_ids"] = sorted(still_wrong["family_id"].tolist())
    result["newly_broken_family_ids"] = sorted(newly_broken["family_id"].tolist())
    return result


def diff_interpretations(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    merged = before.merge(after, on=["family_id", "interp_index"], suffixes=("_before", "_after"))
    return _diff(merged)
