"""Family-aware SIGN evaluation metrics (Phase 5 of
SIGN_GENERALIZATION_PLAN.md). Pure functions over a predictions table
joined with family structure -- no model involved, so this module (and
its tests) can exist and be validated before any model is ever run
against SIGN, ahead of Phase 4/5 actually executing.

Expected input: a DataFrame with at least
`family_id, role ("original"|"interpretation"), predicted_label`
(e.g. `load_sign`'s family table, left-joined with a model's
`predictions.csv` on `example_id`).
"""
from __future__ import annotations

import pandas as pd

ORIGINAL_EXPECTED_LABEL = "sarcastic"
INTERPRETATION_EXPECTED_LABEL = "not_sarcastic"


def add_correctness_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a `correct` bool column: for `role=="original"` rows, correct
    iff predicted sarcastic; for `role=="interpretation"` rows, correct
    iff predicted not_sarcastic."""
    expected = df["role"].map(
        {"original": ORIGINAL_EXPECTED_LABEL, "interpretation": INTERPRETATION_EXPECTED_LABEL}
    )
    out = df.copy()
    out["correct"] = out["predicted_label"] == expected
    return out


def original_sarcasm_detection_rate(df: pd.DataFrame) -> float:
    """Fraction of original (sarcastic) rows correctly predicted sarcastic."""
    df = add_correctness_column(df)
    originals = df[df["role"] == "original"]
    if len(originals) == 0:
        return float("nan")
    return float(originals["correct"].mean())


def interpretation_non_sarcasm_rate(df: pd.DataFrame) -> float:
    """Fraction of interpretation rows correctly predicted not_sarcastic."""
    df = add_correctness_column(df)
    interps = df[df["role"] == "interpretation"]
    if len(interps) == 0:
        return float("nan")
    return float(interps["correct"].mean())


def pairwise_contrastive_accuracy(df: pd.DataFrame) -> float:
    """For every (original, interpretation) pair, success requires the
    original predicted sarcastic AND that interpretation predicted
    not_sarcastic. Returns the fraction of pairs (i.e. of interpretation
    rows) satisfying this."""
    df = add_correctness_column(df)
    originals = df[df["role"] == "original"][["family_id", "correct"]].rename(
        columns={"correct": "original_correct"}
    )
    interps = df[df["role"] == "interpretation"][["family_id", "correct"]].rename(
        columns={"correct": "interp_correct"}
    )
    merged = interps.merge(originals, on="family_id", how="inner")
    if len(merged) == 0:
        return float("nan")
    pair_correct = merged["original_correct"] & merged["interp_correct"]
    return float(pair_correct.mean())


def strict_family_accuracy(df: pd.DataFrame) -> float:
    """Fraction of families where the original AND every interpretation
    row present for that family are all correct."""
    df = add_correctness_column(df)
    family_correct = df.groupby("family_id")["correct"].all()
    if len(family_correct) == 0:
        return float("nan")
    return float(family_correct.mean())


def soft_family_score(df: pd.DataFrame) -> dict:
    """Among families where the original was correctly predicted, the
    mean fraction of that family's interpretations also correctly
    predicted. Returns both the score and the number of families it was
    computed over (families with a wrong original contribute 0 pairs,
    not a 0 score, to avoid conflating "no interpretations right" with
    "original already wrong")."""
    df = add_correctness_column(df)
    originals = df[df["role"] == "original"][["family_id", "correct"]].rename(
        columns={"correct": "original_correct"}
    )
    correct_orig_families = set(originals.loc[originals["original_correct"], "family_id"])
    if not correct_orig_families:
        return {"mean_score": float("nan"), "n_families": 0}

    interps = df[df["role"] == "interpretation"]
    interps = interps[interps["family_id"].isin(correct_orig_families)]
    per_family = interps.groupby("family_id")["correct"].mean()
    return {"mean_score": float(per_family.mean()), "n_families": int(len(per_family))}


def compute_family_metrics(df: pd.DataFrame) -> dict:
    """All Phase-5 metrics in one call, for convenience."""
    return {
        "original_sarcasm_detection_rate": original_sarcasm_detection_rate(df),
        "interpretation_non_sarcasm_rate": interpretation_non_sarcasm_rate(df),
        "pairwise_contrastive_accuracy": pairwise_contrastive_accuracy(df),
        "strict_family_accuracy": strict_family_accuracy(df),
        "soft_family_score": soft_family_score(df),
    }
