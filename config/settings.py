"""Central project settings.

All scripts import paths and API keys from this file instead of hardcoding them.
Create a local `.env` file from `.env.example` before running API-based scripts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    model_outputs_dir: Path = PROJECT_ROOT / "data" / "model_outputs"
    manual_scoring_dir: Path = PROJECT_ROOT / "data" / "manual_scoring"
    summaries_dir: Path = PROJECT_ROOT / "data" / "summaries"
    prompts_dir: Path = PROJECT_ROOT / "prompts"
    results_dir: Path = PROJECT_ROOT / "results"

    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")

    default_openrouter_model: str = "nvidia/nemotron-nano-9b-v2:free"
    default_judge_model: str = "openai/gpt-oss-20b:free"
    default_gemini_model: str = "gemini-2.5-flash-lite"

settings = Settings()
