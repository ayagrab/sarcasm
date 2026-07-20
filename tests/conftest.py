"""Shared fixtures for the test suite. No test here calls a real API,
downloads a model, or touches real project data -- everything uses
temporary fixtures under `tmp_path`."""
from __future__ import annotations
import pandas as pd
import pytest


@pytest.fixture
def sample_model_output_df() -> pd.DataFrame:
    """A tiny, deterministic stand-in for a data/model_outputs/*.csv file."""
    return pd.DataFrame({
        "sarcastic_sentence": ["what a great day", "love this traffic", "best meeting ever"],
        "model_interpretation": ["I am unhappy about the day", "I hate this traffic", "This meeting was terrible"],
        "classification": [3, 3, 1],
    })


@pytest.fixture
def fixture_model_outputs_dir(tmp_path, sample_model_output_df):
    """Builds a tiny data/model_outputs/experiment_0N/*.csv tree, matching
    the real project's layout, for tests that need `load_all_model_outputs`
    or the significance/correlation/linguistic scripts."""
    outputs_dir = tmp_path / "model_outputs"
    for prompt in (1, 2):
        experiment_dir = outputs_dir / f"experiment_{prompt:02d}"
        experiment_dir.mkdir(parents=True)
        for model in ("gemini", "nvidia", "liquid"):
            df = sample_model_output_df.copy()
            df["classification"] = df["classification"] + (prompt - 1)  # vary scores slightly per prompt
            df.to_csv(experiment_dir / f"{model}_run_{prompt:02d}.csv", index=False, encoding="utf-8-sig")
    return outputs_dir
