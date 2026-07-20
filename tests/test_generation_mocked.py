"""Mocked-API tests for the generation scripts.

No real Gemini/OpenRouter call is made -- the model/client factory
functions are monkeypatched with fakes that mimic just enough of the real
SDK response shape to exercise request construction, response parsing, and
the error path.
"""
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from src.generation import generate_with_gemini, generate_with_openrouter


def _write_input(tmp_path: Path) -> Path:
    path = tmp_path / "input.csv"
    pd.DataFrame({"sarcastic_sentence": ["what a great day", "love this traffic"]}).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    return path


class FakeGeminiModel:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def generate_content(self, prompt, safety_settings=None):
        self.calls.append(prompt)
        result = self.responses[len(self.calls) - 1]
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(text=result)


def test_generate_with_gemini_writes_the_model_response_and_uses_the_prompt(tmp_path, monkeypatch):
    input_path = _write_input(tmp_path)
    output_path = tmp_path / "output.csv"
    fake_model = FakeGeminiModel(["I am unhappy about the day", "I hate this traffic"])
    monkeypatch.setattr(generate_with_gemini, "get_gemini_model", lambda: fake_model)

    generate_with_gemini.generate_interpretations(input_path, output_path, sleep_seconds=0)

    assert all("what a great day" in call or "love this traffic" in call for call in fake_model.calls)
    result = pd.read_csv(output_path, encoding="utf-8-sig")
    assert list(result["model_interpretation"]) == ["I am unhappy about the day", "I hate this traffic"]


def test_generate_with_gemini_records_error_on_exception_and_continues(tmp_path, monkeypatch):
    input_path = _write_input(tmp_path)
    output_path = tmp_path / "output.csv"
    fake_model = FakeGeminiModel([RuntimeError("simulated API failure"), "I hate this traffic"])
    monkeypatch.setattr(generate_with_gemini, "get_gemini_model", lambda: fake_model)

    generate_with_gemini.generate_interpretations(input_path, output_path, sleep_seconds=0)

    result = pd.read_csv(output_path, encoding="utf-8-sig")
    assert result.iloc[0]["model_interpretation"] == "ERROR"
    assert result.iloc[1]["model_interpretation"] == "I hate this traffic"


def test_generate_with_gemini_skips_rows_already_completed(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    pd.DataFrame({
        "sarcastic_sentence": ["what a great day", "love this traffic"],
        "model_interpretation": ["already done", ""],
    }).to_csv(input_path, index=False, encoding="utf-8-sig")
    output_path = tmp_path / "output.csv"
    fake_model = FakeGeminiModel(["I hate this traffic"])
    monkeypatch.setattr(generate_with_gemini, "get_gemini_model", lambda: fake_model)

    generate_with_gemini.generate_interpretations(input_path, output_path, sleep_seconds=0)

    assert len(fake_model.calls) == 1  # only the unfinished row was sent
    result = pd.read_csv(output_path, encoding="utf-8-sig")
    assert result.iloc[0]["model_interpretation"] == "already done"
    assert result.iloc[1]["model_interpretation"] == "I hate this traffic"


class FakeOpenRouterClient:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

        class _Completions:
            def create(inner_self, **kwargs):
                self.requests.append(kwargs)
                result = self.responses[len(self.requests) - 1]
                if isinstance(result, Exception):
                    raise result
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result))])

        self.chat = SimpleNamespace(completions=_Completions())


def test_generate_with_openrouter_sends_the_requested_model_and_prompt(tmp_path, monkeypatch):
    input_path = _write_input(tmp_path)
    output_path = tmp_path / "output.csv"
    fake_client = FakeOpenRouterClient(["I am unhappy about the day", "I hate this traffic"])
    monkeypatch.setattr(generate_with_openrouter, "get_openrouter_client", lambda: fake_client)

    generate_with_openrouter.generate_interpretations(
        input_path, output_path, model_id="nvidia/nemotron-nano-9b-v2:free", sleep_seconds=0
    )

    assert all(req["model"] == "nvidia/nemotron-nano-9b-v2:free" for req in fake_client.requests)
    result = pd.read_csv(output_path, encoding="utf-8-sig")
    assert list(result["model_interpretation"]) == ["I am unhappy about the day", "I hate this traffic"]


def test_generate_with_openrouter_records_error_on_exception(tmp_path, monkeypatch):
    input_path = _write_input(tmp_path)
    output_path = tmp_path / "output.csv"
    fake_client = FakeOpenRouterClient([RuntimeError("simulated auth error"), "I hate this traffic"])
    monkeypatch.setattr(generate_with_openrouter, "get_openrouter_client", lambda: fake_client)

    generate_with_openrouter.generate_interpretations(
        input_path, output_path, model_id="nvidia/nemotron-nano-9b-v2:free", sleep_seconds=0
    )

    result = pd.read_csv(output_path, encoding="utf-8-sig")
    assert result.iloc[0]["model_interpretation"] == "ERROR"
