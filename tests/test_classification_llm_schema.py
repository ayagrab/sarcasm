"""Tests for src/classification/llm/schema.py -- structured-output parsing.
No arbitrary label variant is ever accepted (task Section 7)."""
from __future__ import annotations

import json

import pytest

from src.classification.llm.schema import extract_json_object, parse_label_response


def test_extract_json_object_plain():
    assert extract_json_object('{"label": "sarcastic"}') == '{"label": "sarcastic"}'


def test_extract_json_object_markdown_fenced():
    text = '```json\n{"label": "sarcastic"}\n```'
    assert extract_json_object(text) == '{"label": "sarcastic"}'


def test_extract_json_object_with_surrounding_prose():
    text = 'Sure, here it is:\n{"label": "not_sarcastic"}\nHope that helps!'
    assert extract_json_object(text) == '{"label": "not_sarcastic"}'


def test_extract_json_object_raises_when_absent():
    with pytest.raises(ValueError):
        extract_json_object("I cannot help with that.")


def test_parse_label_response_valid_sarcastic():
    result = parse_label_response('{"label": "sarcastic"}')
    assert result["label"] == "sarcastic"


def test_parse_label_response_valid_not_sarcastic():
    result = parse_label_response('{"label": "not_sarcastic"}')
    assert result["label"] == "not_sarcastic"


def test_parse_label_response_normalizes_case_and_hyphen():
    result = parse_label_response('{"label": "Not-Sarcastic"}')
    assert result["label"] == "not_sarcastic"


def test_parse_label_response_preserves_extra_keys():
    result = parse_label_response('{"mismatch_found": true, "label": "sarcastic"}')
    assert result["mismatch_found"] is True
    assert result["label"] == "sarcastic"


def test_parse_label_response_rejects_invalid_label():
    with pytest.raises(ValueError, match="Invalid label"):
        parse_label_response('{"label": "very sarcastic"}')


def test_parse_label_response_rejects_missing_label_key():
    with pytest.raises(ValueError, match="missing 'label'"):
        parse_label_response(json.dumps({"answer": "sarcastic"}))


def test_parse_label_response_rejects_malformed_json():
    with pytest.raises(ValueError):
        parse_label_response("not json at all")
