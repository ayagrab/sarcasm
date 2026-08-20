"""Tests for src/sign/error_analysis/run_error_analysis.py -- Phase 6's
mandatory complete false-negative/false-positive analysis, computed over
hand-constructed prediction+family tables (no real model involved)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.sign.error_analysis.run_error_analysis import (
    METHOD_COLUMNS,
    build_false_negatives,
    build_false_positives,
    cross_model_overlap,
    qualitative_tags,
    quantitative_contrast,
    text_features,
)


@pytest.fixture(scope="module")
def sia():
    from src.sign.characterization.nltk_setup import ensure_nltk_resources

    ensure_nltk_resources()
    from nltk.sentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def _wide_row(family_id, role, interp_index, text, preds):
    row = {"family_id": family_id, "role": role, "interp_index": interp_index, "text": text}
    row.update({f"pred_{m}": p for m, p in zip(["M1", "M2", "M3", "M4", "M5", "M6"], preds)})
    return row


def test_text_features_basic_properties(sia):
    f = text_features("WOW this is great!", sia)
    assert f["has_exclamation_mark"] is True
    assert f["has_all_caps_word"] is True
    assert f["word_length"] == 4

    f2 = text_features("i am stuck on the couch", sia)
    assert f2["has_exclamation_mark"] is False
    assert f2["has_question_mark"] is False


def test_qualitative_tags_flat_delivery_when_no_punctuation_emphasis():
    features = {
        "has_question_mark": False,
        "has_exclamation_mark": False,
        "has_all_caps_word": False,
        "vader_compound": 0.0,
        "word_length": 10,
    }
    tags = qualitative_tags(features)
    assert "flat_delivery_candidate" in tags
    assert "emphatic_delivery_candidate" not in tags


def test_qualitative_tags_short_and_positive():
    features = {
        "has_question_mark": False,
        "has_exclamation_mark": True,
        "has_all_caps_word": False,
        "vader_compound": 0.8,
        "word_length": 3,
    }
    tags = qualitative_tags(features)
    assert "short_ambiguous_candidate" in tags
    assert "positive_surface_wording_candidate" in tags
    assert "emphatic_delivery_candidate" in tags


def test_build_false_negatives_only_includes_originals_missed_by_at_least_one_model(sia):
    wide = pd.DataFrame(
        [
            _wide_row("f1", "original", 0, "original text 1", ["sarcastic"] * 6),  # never missed
            _wide_row("f2", "original", 0, "original text 2", ["not_sarcastic", "sarcastic", "sarcastic", "sarcastic", "sarcastic", "sarcastic"]),  # missed by M1
            _wide_row("f1", "interpretation", 1, "interp 1a", ["not_sarcastic"] * 6),
            _wide_row("f2", "interpretation", 1, "interp 2a", ["not_sarcastic"] * 6),
        ]
    )
    fn = build_false_negatives(wide, sia)
    assert len(fn) == 1
    assert fn.iloc[0]["family_id"] == "f2"
    assert fn.iloc[0]["n_models_missed"] == 1
    assert fn.iloc[0]["which_models_missed"] == "M1"
    assert fn.iloc[0]["interpretation_1"] == "interp 2a"


def test_build_false_positives_only_includes_interpretations_flagged_by_at_least_one_model(sia):
    wide = pd.DataFrame(
        [
            _wide_row("f1", "original", 0, "orig 1", ["sarcastic"] * 6),
            _wide_row("f1", "interpretation", 1, "interp clean", ["not_sarcastic"] * 6),  # never flagged
            _wide_row("f1", "interpretation", 2, "interp flagged", ["sarcastic", "not_sarcastic", "not_sarcastic", "not_sarcastic", "not_sarcastic", "not_sarcastic"]),
        ]
    )
    fp = build_false_positives(wide, sia)
    assert len(fp) == 1
    assert fp.iloc[0]["interp_rank"] == 2
    assert fp.iloc[0]["n_models_flagged_sarcastic"] == 1
    assert fp.iloc[0]["which_models_flagged"] == "M1"
    assert fp.iloc[0]["original_text"] == "orig 1"


def test_quantitative_contrast_separates_ever_missed_from_always_detected(sia):
    wide = pd.DataFrame(
        [
            _wide_row("f1", "original", 0, "short", ["sarcastic"] * 6),  # always detected
            _wide_row("f2", "original", 0, "a much longer original sentence here", ["not_sarcastic"] * 6),  # ever missed
        ]
    )
    contrast = quantitative_contrast(build_false_negatives(wide, sia), wide, sia)
    assert contrast["n_ever_missed"] == 1
    assert contrast["n_always_detected"] == 1
    assert contrast["features"]["word_length"]["ever_missed_mean"] > contrast["features"]["word_length"]["always_detected_mean"]


def test_cross_model_overlap_histogram_and_pairwise():
    wide = pd.DataFrame(
        [
            _wide_row("f1", "original", 0, "o1", ["not_sarcastic"] * 6),  # missed by all 6
            _wide_row("f2", "original", 0, "o2", ["sarcastic"] * 6),  # never missed
            _wide_row("f3", "original", 0, "o3", ["not_sarcastic", "not_sarcastic", "sarcastic", "sarcastic", "sarcastic", "sarcastic"]),  # M1+M2 miss
        ]
    )
    overlap = cross_model_overlap(wide)
    assert overlap["n_total_originals"] == 3
    assert overlap["n_missed_by_all_6"] == 1
    assert overlap["n_never_missed"] == 1
    assert overlap["miss_count_histogram"][2] == 1  # f3 missed by exactly 2 methods
    assert overlap["pairwise_overlap"]["M1_vs_M2"]["overlap_count"] == 2  # f1 and f3 both missed by M1 and M2
