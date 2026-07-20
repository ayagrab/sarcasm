"""Mocked-API tests for src/evaluation/evaluate_with_llm.py (the LLM judge).

No real OpenRouter call is made. `get_openrouter_client` is monkeypatched
with a fake client whose `chat.completions.create` returns a scripted
response, to exercise response parsing, count/range validation, and the
retry-then-raise error path.
"""
from types import SimpleNamespace
import pandas as pd
import pytest
from src.evaluation import evaluate_with_llm


def _fake_client(contents):
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=contents[len(calls) - 1]))])

    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions())), calls


def _batch_df():
    return pd.DataFrame({
        "sarcastic_sentence": ["what a great day", "love this traffic"],
        "model_interpretation": ["I am unhappy about the day", "I hate this traffic"],
    })


def test_classify_batch_parses_a_valid_response(monkeypatch):
    client, calls = _fake_client(['[{"score": 3}, {"score": 1}]'])
    monkeypatch.setattr(evaluate_with_llm, "get_openrouter_client", lambda: client)

    scores = evaluate_with_llm.classify_batch(_batch_df(), model_id="test-model")

    assert scores == [3, 1]
    assert calls[0]["model"] == "test-model"


def test_classify_batch_parses_a_markdown_fenced_response(monkeypatch):
    client, _ = _fake_client(['```json\n[{"score": 2}, {"score": 2}]\n```'])
    monkeypatch.setattr(evaluate_with_llm, "get_openrouter_client", lambda: client)

    scores = evaluate_with_llm.classify_batch(_batch_df(), model_id="test-model")
    assert scores == [2, 2]


def test_classify_batch_retries_then_raises_on_malformed_json(monkeypatch):
    client, calls = _fake_client(["not json at all", "still not json"])
    monkeypatch.setattr(evaluate_with_llm, "get_openrouter_client", lambda: client)

    with pytest.raises(Exception):
        evaluate_with_llm.classify_batch(_batch_df(), model_id="test-model", max_retries=2, wait_seconds=0)
    assert len(calls) == 2


def test_classify_batch_retries_then_raises_on_wrong_score_count(monkeypatch):
    client, calls = _fake_client(['[{"score": 3}]', '[{"score": 3}]'])  # only 1 score for a 2-row batch
    monkeypatch.setattr(evaluate_with_llm, "get_openrouter_client", lambda: client)

    with pytest.raises(Exception):
        evaluate_with_llm.classify_batch(_batch_df(), model_id="test-model", max_retries=2, wait_seconds=0)
    assert len(calls) == 2


def test_classify_batch_retries_then_raises_on_out_of_range_score(monkeypatch):
    client, calls = _fake_client(['[{"score": 5}, {"score": 1}]', '[{"score": 5}, {"score": 1}]'])
    monkeypatch.setattr(evaluate_with_llm, "get_openrouter_client", lambda: client)

    with pytest.raises(Exception):
        evaluate_with_llm.classify_batch(_batch_df(), model_id="test-model", max_retries=2, wait_seconds=0)
    assert len(calls) == 2


def test_classify_batch_succeeds_after_one_retry(monkeypatch):
    client, calls = _fake_client(["not json", '[{"score": 3}, {"score": 1}]'])
    monkeypatch.setattr(evaluate_with_llm, "get_openrouter_client", lambda: client)

    scores = evaluate_with_llm.classify_batch(_batch_df(), model_id="test-model", max_retries=3, wait_seconds=0)
    assert scores == [3, 1]
    assert len(calls) == 2


def test_evaluate_file_only_evaluates_unclassified_rows(tmp_path, monkeypatch):
    input_path = tmp_path / "output.csv"
    pd.DataFrame({
        "sarcastic_sentence": ["what a great day", "love this traffic"],
        "model_interpretation": ["I am unhappy about the day", "I hate this traffic"],
        "classification": [3, ""],
    }).to_csv(input_path, index=False, encoding="utf-8-sig")

    client, calls = _fake_client(['[{"score": 2}]'])  # only 1 unclassified row
    monkeypatch.setattr(evaluate_with_llm, "get_openrouter_client", lambda: client)

    evaluate_with_llm.evaluate_file(input_path, model_id="test-model", batch_size=10)

    result = pd.read_csv(input_path, encoding="utf-8-sig")
    assert list(result["classification"]) == [3, 2]
    assert len(calls) == 1
