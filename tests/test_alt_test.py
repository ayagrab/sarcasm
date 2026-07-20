"""Tests for src/common/alt_test.py.

`run_alt_test.py` was already manually verified against the project's real
`data/alt_test/*.json` to reproduce the documented result exactly (Winning
Rate 0.67, Advantage Probability 0.77). These tests instead check the
algorithm's behavior on small, constructed, deterministic scenarios.
"""
import random
import pytest
from src.common.alt_test import accuracy, neg_rmse, sim, alt_test


def test_accuracy_scoring_function():
    assert accuracy("A", ["A", "A", "B"]) == pytest.approx(2 / 3)
    assert accuracy("A", ["B", "B"]) == 0.0
    assert accuracy("A", ["A"]) == 1.0


def test_neg_rmse_scoring_function():
    assert neg_rmse(3, [3, 3]) == 0.0
    assert neg_rmse(1, [3]) == pytest.approx(-2.0)


def test_sim_scoring_function_uses_the_given_similarity_function():
    result = sim("cat", ["cat", "dog"], similarity_func=lambda a, b: 1.0 if a == b else 0.0)
    assert result == pytest.approx(0.5)


def _build_categorical_dataset(n_instances: int, n_errors_per_human: int, llm_perfect: bool, seed: int):
    """3 human annotators labeling A/B instances against a hidden ground
    truth, each with a few random errors; an LLM that is either perfect or
    always wrong relative to the ground truth."""
    rng = random.Random(seed)
    truth = {f"i{i}": ("A" if i % 2 == 0 else "B") for i in range(n_instances)}
    humans_annotations = {}
    for human in ("h1", "h2", "h3"):
        error_instances = set(rng.sample(list(truth), n_errors_per_human))
        humans_annotations[human] = {
            instance: (("B" if value == "A" else "A") if instance in error_instances else value)
            for instance, value in truth.items()
        }
    if llm_perfect:
        llm_annotations = dict(truth)
    else:
        llm_annotations = {instance: ("B" if value == "A" else "A") for instance, value in truth.items()}
    return humans_annotations, llm_annotations


def test_a_perfect_llm_beats_noisy_human_annotators():
    humans_annotations, llm_annotations = _build_categorical_dataset(
        n_instances=40, n_errors_per_human=6, llm_perfect=True, seed=42
    )
    winning_rate, advantage_prob = alt_test(
        llm_annotations, humans_annotations, scoring_function="accuracy",
        epsilon=0.2, min_instances_per_human=10,
    )
    assert 0.0 <= winning_rate <= 1.0
    assert 0.0 <= advantage_prob <= 1.0
    assert winning_rate >= 0.5
    assert advantage_prob > 0.8


def test_a_consistently_wrong_llm_loses_to_accurate_human_annotators():
    humans_annotations, llm_annotations = _build_categorical_dataset(
        n_instances=40, n_errors_per_human=2, llm_perfect=False, seed=7
    )
    winning_rate, advantage_prob = alt_test(
        llm_annotations, humans_annotations, scoring_function="accuracy",
        epsilon=0.2, min_instances_per_human=10,
    )
    assert winning_rate == 0.0
    assert advantage_prob < 0.5


def test_instances_with_too_few_human_annotators_are_dropped(capsys):
    humans_annotations = {
        "h1": {"i1": "A", "i2": "B", "i3": "A", "i4": "B", "i5": "A"},
        "h2": {"i1": "A", "i2": "B", "i3": "B", "i4": "B", "i5": "A"},
        # i5 is only labeled by h1 and h2, not h3 -- still >= 2, kept.
        # i6 is only labeled by h1 -- below min_humans_per_instance, dropped.
        "h3": {"i1": "A", "i2": "A", "i3": "A", "i4": "B"},
    }
    humans_annotations["h1"]["i6"] = "A"
    llm_annotations = {"i1": "A", "i2": "B", "i3": "A", "i4": "B", "i5": "A", "i6": "A"}

    winning_rate, advantage_prob = alt_test(
        llm_annotations, humans_annotations, scoring_function="accuracy",
        min_humans_per_instance=2, min_instances_per_human=3,
    )
    captured = capsys.readouterr()
    assert "Dropped 1 instances with less than 2 annotators." in captured.out
    assert 0.0 <= winning_rate <= 1.0


def test_annotators_below_min_instances_are_skipped(capsys):
    humans_annotations = {
        "h1": {"i1": "A", "i2": "B", "i3": "A", "i4": "B", "i5": "A"},
        "h2": {"i1": "A", "i2": "B", "i3": "A", "i4": "B", "i5": "A"},
        "h3": {"i1": "A"},  # only 1 instance -- should be skipped
    }
    llm_annotations = {"i1": "A", "i2": "B", "i3": "A", "i4": "B", "i5": "A"}

    alt_test(llm_annotations, humans_annotations, scoring_function="accuracy", min_instances_per_human=3)
    captured = capsys.readouterr()
    assert "Skipping annotator h3" in captured.out
