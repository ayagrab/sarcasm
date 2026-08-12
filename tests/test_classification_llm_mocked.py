"""Mocked-API tests for src/classification/llm/run_llm_classification.py.

No real OpenRouter call is made -- `get_llm_client` is monkeypatched with a
fake client whose `chat.completions.create` returns a scripted response,
matching this repository's existing convention for API-backed code (see
tests/test_evaluate_with_llm_mocked.py). `run_llm_experiment` is exercised
against a tiny fixture split directory (monkeypatched settings), not the
real data/splits/.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.classification.llm import run_llm_classification as llm_mod


def _fake_client(contents):
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            content = contents[len(calls) - 1]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=None,
            )

    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions())), calls


def test_classify_one_parses_a_valid_response(monkeypatch, tmp_path):
    client, calls = _fake_client(['{"label": "sarcastic"}'])
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path / "cache")

    result = llm_mod.classify_one(client, "test-model", "zero_shot", "oh great", use_cache=False)

    assert result["label"] == "sarcastic"
    assert calls[0]["model"] == "test-model"


def test_classify_one_retries_then_succeeds(monkeypatch, tmp_path):
    client, calls = _fake_client(["not json", '{"label": "not_sarcastic"}'])
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path / "cache")

    result = llm_mod.classify_one(
        client, "test-model", "zero_shot", "meh", max_retries=3, wait_seconds=0, use_cache=False
    )

    assert result["label"] == "not_sarcastic"
    assert len(calls) == 2


def test_classify_one_retries_then_raises_on_invalid_label(monkeypatch, tmp_path):
    client, calls = _fake_client(['{"label": "kinda sarcastic"}', '{"label": "kinda sarcastic"}'])
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path / "cache")

    with pytest.raises(RuntimeError):
        llm_mod.classify_one(
            client, "test-model", "zero_shot", "meh", max_retries=2, wait_seconds=0, use_cache=False
        )
    assert len(calls) == 2


def test_classify_one_uses_cache_on_second_call(monkeypatch, tmp_path):
    client, calls = _fake_client(['{"label": "sarcastic"}'])
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path / "cache")

    llm_mod.classify_one(client, "test-model", "zero_shot", "oh great", use_cache=True)
    llm_mod.classify_one(client, "test-model", "zero_shot", "oh great", use_cache=True)

    assert len(calls) == 1  # second call served from disk cache


def test_build_messages_few_shot_substitutes_demonstrations():
    messages = llm_mod.build_messages("few_shot", "test sentence", demonstrations="DEMO_BLOCK")
    assert "DEMO_BLOCK" in messages[0]["content"]
    assert "test sentence" in messages[1]["content"]


@pytest.fixture
def fixture_splits(tmp_path):
    train_df = pd.DataFrame(
        {
            "example_id": [f"T-{i}" for i in range(8)],
            "category": ["GEN"] * 8,
            "label": (["sarcastic", "not_sarcastic"] * 4),
            "text": [f"train sentence {i}" for i in range(8)],
        }
    )
    dev_df = pd.DataFrame(
        {
            "example_id": ["D-0", "D-1", "D-2", "D-3"],
            "category": ["GEN"] * 4,
            "label": ["sarcastic", "not_sarcastic", "sarcastic", "not_sarcastic"],
            "text": ["dev sentence 0", "dev sentence 1", "dev sentence 2", "dev sentence 3"],
        }
    )
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    train_df.to_csv(splits_dir / "train.csv", index=False)
    dev_df.to_csv(splits_dir / "dev.csv", index=False)
    return splits_dir


def test_run_llm_experiment_zero_shot_end_to_end(monkeypatch, tmp_path, fixture_splits):
    client, calls = _fake_client(['{"label": "sarcastic"}'] * 4)
    monkeypatch.setattr(llm_mod, "get_llm_client", lambda provider="openrouter": client)
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path / "cache")

    result = llm_mod.run_llm_experiment(
        experiment_id="TEST-001",
        mode="zero_shot",
        eval_split="dev",
        model="fake-model",
        concurrency=1,
        use_cache=False,
        splits_dir=fixture_splits,
        results_dir=tmp_path / "results",
    )

    assert result["metrics"]["n_examples"] == 4
    assert len(calls) == 4
    predictions = pd.read_csv(tmp_path / "results" / "TEST-001" / "predictions.csv")
    assert set(predictions["predicted_label"]) == {"sarcastic"}


def test_run_llm_experiment_few_shot_records_demo_example_ids(monkeypatch, tmp_path, fixture_splits):
    client, calls = _fake_client(['{"label": "not_sarcastic"}'] * 4)
    monkeypatch.setattr(llm_mod, "get_llm_client", lambda provider="openrouter": client)
    monkeypatch.setattr(llm_mod, "CACHE_DIR", tmp_path / "cache")

    result = llm_mod.run_llm_experiment(
        experiment_id="TEST-002",
        mode="few_shot",
        eval_split="dev",
        model="fake-model",
        few_shot_variant="random",
        n_shots=4,
        concurrency=1,
        use_cache=False,
        splits_dir=fixture_splits,
        results_dir=tmp_path / "results",
    )

    assert result["config"]["demo_example_ids"]
    assert all(eid.startswith("T-") for eid in result["config"]["demo_example_ids"])
    # demonstrations must come from TRAIN only, never dev/test
    assert not any(eid.startswith("D-") for eid in result["config"]["demo_example_ids"])
