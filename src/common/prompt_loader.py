"""Utilities for loading prompt templates from the prompts/ directory."""

from __future__ import annotations

from pathlib import Path

from config.settings import settings


def load_prompt(relative_path: str, prompts_dir: Path | None = None) -> str:
    """
    Load a prompt template from the project's prompts directory.

    Parameters
    ----------
    relative_path : str
        Relative path to the prompt file inside the prompts directory.

        Examples
        --------
        generation/generation_prompt_v1.txt
        generation/generation_prompt_v4.txt
        evaluation/llm_judge_3_level_prompt.txt

    prompts_dir : Path | None
        Optional prompts directory. If not provided, the project's default
        prompts directory defined in settings is used.

    Returns
    -------
    str
        The full prompt text.

    Raises
    ------
    FileNotFoundError
        If the prompt file does not exist.
    """

    prompts_root = prompts_dir or settings.prompts_dir
    prompt_path = prompts_root / relative_path

    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Prompt file not found:\n{prompt_path}"
        )

    return prompt_path.read_text(encoding="utf-8")