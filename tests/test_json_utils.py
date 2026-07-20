"""Tests for src/common/json_utils.py -- used to parse the LLM judge's
batch score responses."""
import pytest
from src.common.json_utils import extract_json_array


def test_extracts_a_plain_json_array():
    assert extract_json_array('[{"score": 1}, {"score": 2}]') == '[{"score": 1}, {"score": 2}]'


def test_extracts_json_wrapped_in_markdown_fences():
    text = '```json\n[{"score": 3}]\n```'
    assert extract_json_array(text) == '[{"score": 3}]'


def test_extracts_json_wrapped_in_plain_fences():
    text = '```\n[{"score": 1}]\n```'
    assert extract_json_array(text) == '[{"score": 1}]'


def test_extracts_array_even_with_surrounding_prose():
    text = 'Here are the scores:\n[{"score": 2}, {"score": 3}]\nThanks!'
    assert extract_json_array(text) == '[{"score": 2}, {"score": 3}]'


def test_raises_value_error_when_no_array_present():
    with pytest.raises(ValueError):
        extract_json_array("I cannot help with that request.")
