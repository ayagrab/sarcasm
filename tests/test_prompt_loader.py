"""Tests for src/common/prompt_loader.py.

These tests only READ the real prompt files to confirm they load correctly
-- they never write to or modify any file under prompts/.
"""
from pathlib import Path
import pytest
from src.common.prompt_loader import load_prompt
from config.settings import settings

REAL_GENERATION_PROMPTS = [
    "generation/generation_prompt_v1.txt",
    "generation/generation_prompt_v2.txt",
    "generation/generation_prompt_v3.txt",
    "generation/generation_prompt_v4.txt",
]


@pytest.mark.parametrize("relative_path", REAL_GENERATION_PROMPTS)
def test_real_generation_prompts_load_and_use_the_shared_placeholder(relative_path):
    text = load_prompt(relative_path)
    assert text.strip() != ""
    assert "{sarcastic_sentence}" in text
    # formatting must not raise -- confirms the placeholder name is consistent
    formatted = text.format(sarcastic_sentence="a test sentence")
    assert "a test sentence" in formatted


def test_real_llm_judge_prompt_loads():
    text = load_prompt("evaluation/llm_judge_prompt.txt")
    assert text.strip() != ""


def test_real_nli_templates_load_and_format():
    premise = load_prompt("evaluation/nli_premise_template.txt")
    hypothesis = load_prompt("evaluation/nli_hypothesis_template.txt")
    assert premise.format(sarcastic_sentence="X") == "X"
    assert hypothesis.format(model_interpretation="Y") == "Y"


def test_missing_prompt_file_raises_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        load_prompt("generation/this_file_does_not_exist.txt")


def test_custom_prompts_dir_is_used_when_given(tmp_path: Path):
    custom_dir = tmp_path / "custom_prompts"
    (custom_dir / "sub").mkdir(parents=True)
    (custom_dir / "sub" / "example.txt").write_text("hello {name}", encoding="utf-8")

    text = load_prompt("sub/example.txt", prompts_dir=custom_dir)
    assert text == "hello {name}"


def test_default_prompts_dir_matches_settings():
    assert settings.prompts_dir.is_dir()
