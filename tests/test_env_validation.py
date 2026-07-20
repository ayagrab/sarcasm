"""Tests that API-backed components fail with a clear, human-readable error
when the required API key is missing -- instead of a confusing traceback.

Each test replaces the module-local `settings` reference with a fake object
(the real `Settings` is a frozen dataclass and cannot be mutated in place),
so no real environment variables or `.env` file are touched.
"""
from types import SimpleNamespace
import pytest


def test_get_gemini_model_raises_a_clear_error_when_key_missing(monkeypatch):
    from src.common import gemini_client
    monkeypatch.setattr(gemini_client, "settings", SimpleNamespace(gemini_api_key=None, default_gemini_model="x"))
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is missing"):
        gemini_client.get_gemini_model()


def test_get_openrouter_client_raises_a_clear_error_when_key_missing(monkeypatch):
    from src.common import openrouter_client
    monkeypatch.setattr(openrouter_client, "settings", SimpleNamespace(openrouter_api_key=None))
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is missing"):
        openrouter_client.get_openrouter_client()


def test_get_openrouter_client_raises_for_empty_string_key_too(monkeypatch):
    from src.common import openrouter_client
    monkeypatch.setattr(openrouter_client, "settings", SimpleNamespace(openrouter_api_key=""))
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is missing"):
        openrouter_client.get_openrouter_client()


def test_check_openrouter_limit_raises_a_clear_error_when_key_missing(monkeypatch):
    from src.tools import check_openrouter_limit
    monkeypatch.setattr(check_openrouter_limit, "settings", SimpleNamespace(openrouter_api_key=None))
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is missing"):
        check_openrouter_limit.main()
