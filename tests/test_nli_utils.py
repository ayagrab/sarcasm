"""Tests for src/common/nli_utils.py -- entailment label mapping.

None of these tests load an NLI model; they use hand-built id2label dicts
and probability lists to verify the decision logic in isolation.
"""
import pytest
from src.common.nli_utils import classify_entailment, resolve_label_index, normalize_id2label


STANDARD_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def test_entailment_wins():
    assert classify_entailment([0.7, 0.2, 0.1], STANDARD_LABELS) == 1


def test_contradiction_wins():
    assert classify_entailment([0.1, 0.2, 0.7], STANDARD_LABELS) == 0


def test_neutral_does_not_affect_the_decision():
    # neutral has the highest probability, but the decision is entailment vs. contradiction only
    assert classify_entailment([0.4, 0.6, 0.0], STANDARD_LABELS) == 1


def test_case_insensitive_label_names():
    labels = {0: "ENTAILMENT", 1: "Neutral", 2: "CONTRADICTION"}
    assert classify_entailment([0.9, 0.05, 0.05], labels) == 1


def test_labels_in_a_different_order():
    labels = {0: "contradiction", 1: "entailment", 2: "neutral"}
    assert classify_entailment([0.1, 0.8, 0.1], labels) == 1
    assert classify_entailment([0.8, 0.1, 0.1], labels) == 0


def test_generic_label_placeholders_use_documented_fallback():
    # Some model configs expose "LABEL_0"/"LABEL_1"/"LABEL_2" instead of
    # real names; classify_entailment should not silently guess -- it uses
    # the explicit, documented fallback mapping for the project's default model.
    labels = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    assert classify_entailment([0.9, 0.05, 0.05], labels) == 1  # index 0 = entailment per fallback
    assert classify_entailment([0.05, 0.05, 0.9], labels) == 0  # index 2 = contradiction per fallback


def test_unresolvable_labels_raise_a_clear_error_without_fallback():
    with pytest.raises(ValueError, match="Could not resolve"):
        classify_entailment([0.5, 0.5], {0: "LABEL_0", 1: "LABEL_1"}, fallback=None)


def test_resolve_label_index_returns_none_when_not_found():
    assert resolve_label_index({0: "foo", 1: "bar"}, "entailment", fallback=None) is None


def test_normalize_id2label_lowercases_and_strips():
    assert normalize_id2label({0: " Entailment ", 1: "NEUTRAL"}) == {"entailment": 0, "neutral": 1}
