"""Tests for src/sign/domain_adaptation/error_diff.py -- Phase 8's
before/after (zero-transfer vs. adapted) error diff."""
from __future__ import annotations

import pandas as pd

from src.sign.domain_adaptation.error_diff import (
    diff_originals,
    interpretations_correctness,
    originals_correctness,
)


def _family_table():
    rows = [
        {"example_id": "f1-orig", "family_id": "f1", "role": "original", "interp_index": 0},
        {"example_id": "f2-orig", "family_id": "f2", "role": "original", "interp_index": 0},
        {"example_id": "f3-orig", "family_id": "f3", "role": "original", "interp_index": 0},
        {"example_id": "f1-i1", "family_id": "f1", "role": "interpretation", "interp_index": 1},
    ]
    return pd.DataFrame(rows)


def _preds(rows):
    return pd.DataFrame(rows, columns=["example_id", "gold_label", "predicted_label"])


def test_originals_correctness_flags_sarcastic_predictions_correct():
    family_table = _family_table()
    predictions = _preds(
        [
            ("f1-orig", "sarcastic", "sarcastic"),
            ("f2-orig", "sarcastic", "not_sarcastic"),
        ]
    )
    correctness = originals_correctness(predictions, family_table)
    assert dict(zip(correctness["family_id"], correctness["correct"])) == {"f1": True, "f2": False}


def test_diff_originals_classifies_fixed_still_wrong_newly_broken():
    before = pd.DataFrame(
        {"family_id": ["f1", "f2", "f3"], "correct": [False, False, True]}
    )
    after = pd.DataFrame(
        {"family_id": ["f1", "f2", "f3"], "correct": [True, False, False]}
    )
    result = diff_originals(before, after)
    assert result["n_fixed"] == 1
    assert result["fixed_family_ids"] == ["f1"]
    assert result["n_still_wrong"] == 1
    assert result["still_wrong_family_ids"] == ["f2"]
    assert result["n_newly_broken"] == 1
    assert result["newly_broken_family_ids"] == ["f3"]
    assert result["n_still_correct"] == 0


def test_interpretations_correctness_flags_not_sarcastic_predictions_correct():
    family_table = _family_table()
    predictions = _preds([("f1-i1", "not_sarcastic", "not_sarcastic")])
    correctness = interpretations_correctness(predictions, family_table)
    assert correctness.iloc[0]["correct"] == True
    assert correctness.iloc[0]["interp_index"] == 1
