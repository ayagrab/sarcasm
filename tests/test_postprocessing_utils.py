"""Tests for smaller postprocessing helpers, using tiny synthetic fixtures
(never the real project data)."""
import json
from pathlib import Path
import pandas as pd
import pytest
from src.postprocessing.create_manual_sample import find_one_csv
from src.postprocessing.human_llm_agreement import (
    _parse_instance_key, load_human_scores, load_llm_scores,
    attach_interpretations, build_comparison_table, compute_fleiss_kappa,
)
from src.postprocessing.extract_case_studies import find_perfect_agreement_cases, find_discrepancy_cases


def test_find_one_csv_returns_the_single_match(tmp_path: Path):
    (tmp_path / "gemini_run_01.csv").write_text("a,b\n1,2\n")
    result = find_one_csv(tmp_path, "gemini")
    assert result.name == "gemini_run_01.csv"


def test_find_one_csv_raises_when_no_match(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_one_csv(tmp_path, "gemini")


def test_find_one_csv_raises_when_multiple_matches(tmp_path: Path):
    (tmp_path / "gemini_run_01.csv").write_text("a,b\n1,2\n")
    (tmp_path / "gemini_run_02.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        find_one_csv(tmp_path, "gemini")


def test_parse_instance_key_splits_folder_model_and_sentence():
    folder, model, sentence = _parse_instance_key("1_gemini_this is a sarcastic tweet")
    assert (folder, model, sentence) == ("1", "gemini", "this is a sarcastic tweet")


def _write_alt_test_fixture(tmp_path: Path) -> tuple[Path, Path]:
    humans = {
        "annA": {"1_gemini_hello world": 3, "1_gemini_another one": 1},
        "annB": {"1_gemini_hello world": 2, "1_gemini_another one": 1},
        "annC": {"1_gemini_hello world": 3, "1_gemini_another one": 2},
    }
    llm = {"1_gemini_hello world": 3, "1_gemini_another one": 1}
    humans_path = tmp_path / "humans_annotations.json"
    llm_path = tmp_path / "llm_annotations.json"
    humans_path.write_text(json.dumps(humans), encoding="utf-8")
    llm_path.write_text(json.dumps(llm), encoding="utf-8")
    return humans_path, llm_path


def test_load_human_scores_produces_long_format(tmp_path: Path):
    humans_path, _ = _write_alt_test_fixture(tmp_path)
    df = load_human_scores(humans_path)
    assert len(df) == 6  # 3 annotators * 2 instances
    assert set(df.columns) == {"human", "folder", "model", "sentence", "score"}


def test_load_llm_scores_produces_one_row_per_instance(tmp_path: Path):
    _, llm_path = _write_alt_test_fixture(tmp_path)
    df = load_llm_scores(llm_path)
    assert len(df) == 2
    assert set(df.columns) == {"folder", "model", "sentence", "score"}


def test_build_comparison_table_merges_llm_and_average_human_score(tmp_path: Path):
    humans_path, llm_path = _write_alt_test_fixture(tmp_path)
    merged = build_comparison_table(humans_path, llm_path)
    assert len(merged) == 2
    hello_row = merged[merged["sentence"] == "hello world"].iloc[0]
    assert hello_row["human_score"] == pytest.approx((3 + 2 + 3) / 3)
    assert hello_row["chatgpt_score"] == 3
    assert hello_row["human_score_rounded"] == round((3 + 2 + 3) / 3)


def test_compute_fleiss_kappa_returns_a_float_in_range(tmp_path: Path):
    humans_path, _ = _write_alt_test_fixture(tmp_path)
    human_df = load_human_scores(humans_path)
    kappa = compute_fleiss_kappa(human_df)
    assert -1.0 <= kappa <= 1.0


def test_attach_interpretations_looks_up_text_from_model_outputs(tmp_path: Path):
    outputs_dir = tmp_path / "model_outputs" / "experiment_01"
    outputs_dir.mkdir(parents=True)
    pd.DataFrame({
        "sarcastic_sentence": ["hello world"],
        "model_interpretation": ["a sincere hello"],
        "classification": [3],
    }).to_csv(outputs_dir / "gemini_run_01.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame({"folder": ["1"], "model": ["gemini"], "sentence": ["hello world"]})
    result = attach_interpretations(df, tmp_path / "model_outputs")
    assert result.iloc[0]["model_interpretation"] == "a sincere hello"


def test_find_perfect_agreement_cases_filters_correctly():
    df = pd.DataFrame({
        "human_score_rounded": [3, 2, 1],
        "chatgpt_score": [3, 1, 1],
    })
    agreement = find_perfect_agreement_cases(df)
    assert len(agreement) == 2


def test_find_discrepancy_cases_returns_the_largest_gaps_first():
    df = pd.DataFrame({
        "human_score": [3.0, 1.0, 2.0],
        "chatgpt_score": [1, 1, 2],
    })
    discrepancies = find_discrepancy_cases(df, top_n=2)
    assert len(discrepancies) == 2
    assert discrepancies.iloc[0]["human_score"] == 3.0  # biggest gap (|3-1|=2) first
