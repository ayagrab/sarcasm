"""Tests for src/common/file_utils.py."""
from pathlib import Path
import pandas as pd
from src.common.file_utils import ensure_parent_dir, read_csv_flexible, save_csv, load_all_model_outputs


def test_ensure_parent_dir_creates_missing_directories(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c.csv"
    assert not target.parent.exists()
    ensure_parent_dir(target)
    assert target.parent.is_dir()


def test_read_csv_flexible_with_a_normal_header(tmp_path: Path):
    path = tmp_path / "normal.csv"
    path.write_text("sarcastic_sentence,model_interpretation\nhi,bye\n", encoding="utf-8-sig")
    df = read_csv_flexible(path, ["sarcastic_sentence", "model_interpretation"])
    assert list(df.columns) == ["sarcastic_sentence", "model_interpretation"]
    assert df.iloc[0]["sarcastic_sentence"] == "hi"


def test_read_csv_flexible_with_a_headerless_file(tmp_path: Path):
    # matches the original SIGN-paper dataset format: no header row at all
    path = tmp_path / "headerless.csv"
    path.write_text("hi there,extra,columns\nsecond row,x,y\n", encoding="utf-8-sig")
    df = read_csv_flexible(path, ["sarcastic_sentence", "model_interpretation"])
    assert list(df.columns) == ["sarcastic_sentence", "model_interpretation"]
    assert df.iloc[0]["sarcastic_sentence"] == "hi there"


def test_save_csv_round_trips_and_creates_directories(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "nested" / "out.csv"
    save_csv(df, path)
    reloaded = pd.read_csv(path, encoding="utf-8-sig")
    pd.testing.assert_frame_equal(reloaded, df)


def test_load_all_model_outputs_adds_prompt_and_model_columns(fixture_model_outputs_dir):
    df = load_all_model_outputs(fixture_model_outputs_dir)
    # 2 prompts * 3 models * 3 rows = 18 rows
    assert len(df) == 18
    assert set(df["prompt"].unique()) == {1, 2}
    assert set(df["model"].unique()) == {"gemini", "nvidia", "liquid"}
    assert df["classification"].dtype.kind in "if"  # numeric (int or float)


def test_load_all_model_outputs_drops_non_numeric_classification(tmp_path: Path):
    outputs_dir = tmp_path / "model_outputs"
    experiment_dir = outputs_dir / "experiment_01"
    experiment_dir.mkdir(parents=True)
    pd.DataFrame({
        "sarcastic_sentence": ["a", "b"],
        "model_interpretation": ["x", "y"],
        "classification": ["<unset>", "3"],
    }).to_csv(experiment_dir / "gemini_run_01.csv", index=False, encoding="utf-8-sig")

    df = load_all_model_outputs(outputs_dir)
    assert len(df) == 1
    assert df.iloc[0]["classification"] == 3.0
