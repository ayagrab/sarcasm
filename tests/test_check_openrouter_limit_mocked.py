"""Mocked-API tests for src/tools/check_openrouter_limit.py.

No real HTTP request is made -- `requests.get` is monkeypatched.
"""
from types import SimpleNamespace
import pytest
import requests
from src.tools import check_openrouter_limit


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def test_main_prints_the_key_usage_on_success(monkeypatch, capsys):
    monkeypatch.setattr(check_openrouter_limit, "settings", SimpleNamespace(openrouter_api_key="fake-key"))
    monkeypatch.setattr(
        requests, "get",
        lambda url, headers, timeout: _FakeResponse({"data": {"limit": 10, "usage": 1.5}}),
    )

    check_openrouter_limit.main()

    captured = capsys.readouterr()
    assert '"limit": 10' in captured.out
    assert '"usage": 1.5' in captured.out


def test_main_raises_http_error_on_authentication_failure(monkeypatch):
    monkeypatch.setattr(check_openrouter_limit, "settings", SimpleNamespace(openrouter_api_key="invalid-key"))
    monkeypatch.setattr(
        requests, "get",
        lambda url, headers, timeout: _FakeResponse({"error": "unauthorized"}, status_code=401),
    )

    with pytest.raises(requests.HTTPError):
        check_openrouter_limit.main()
