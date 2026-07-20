"""Tests for src/postprocessing/calculate_text_metrics.py."""
from pathlib import Path
import pandas as pd
from src.postprocessing.calculate_text_metrics import calculate_pinc, sigmoid, simple_ngram_recall, calculate_metrics


def test_calculate_pinc_all_new_words_is_one():
    assert calculate_pinc("the cat sat", "a dog ran") == 1.0


def test_calculate_pinc_identical_text_is_zero():
    assert calculate_pinc("the cat sat", "the cat sat") == 0.0


def test_calculate_pinc_empty_interpretation_is_zero():
    assert calculate_pinc("the cat sat", "") == 0.0


def test_sigmoid_at_zero_is_one_half():
    assert abs(sigmoid(0.0) - 0.5) < 1e-9


def test_simple_ngram_recall_full_overlap_unigram():
    assert simple_ngram_recall("the cat sat", "the cat sat", n=1) == 1.0


def test_simple_ngram_recall_no_overlap():
    assert simple_ngram_recall("the cat sat", "a dog ran", n=1) == 0.0


def test_calculate_metrics_on_a_tiny_fixture(tmp_path: Path):
    input_path = tmp_path / "tiny.csv"
    pd.DataFrame({
        "sarcastic_sentence": ["what a great day", "love this traffic"],
        "model_interpretation": ["I am unhappy about the day", "I hate this traffic"],
    }).to_csv(input_path, index=False, encoding="utf-8-sig")

    summary = calculate_metrics(input_path)
    assert summary.shape[0] == 1
    row = summary.iloc[0]
    assert row["rows"] == 2
    for column in ["avg_bleu", "avg_rouge1_recall", "avg_rouge2_recall", "avg_pinc", "avg_combined_pinc_sigmoid_bleu"]:
        assert 0.0 <= row[column] <= 1.0
