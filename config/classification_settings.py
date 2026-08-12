"""Settings for the sarcasm *classification/detection* phase.

Kept separate from `config/settings.py`, which is scoped to the existing
sarcasm *interpretation* pipeline (generation/evaluation of sincere
rewrites), so the two phases never share or accidentally overwrite each
other's configuration. See `EXPERIMENT_LOG.md` for why the two phases are
kept apart.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class ClassificationSettings:
    project_root: Path = PROJECT_ROOT

    # Data
    sarcasm_v2_raw_dir: Path = PROJECT_ROOT / "data" / "raw" / "sarcasm_corpus_v2"
    canonical_dataset_path: Path = PROJECT_ROOT / "data" / "processed" / "sarcasm_v2_canonical.csv"
    audit_report_path: Path = PROJECT_ROOT / "data" / "processed" / "sarcasm_v2_audit_report.json"
    splits_dir: Path = PROJECT_ROOT / "data" / "splits"
    split_assignments_path: Path = PROJECT_ROOT / "data" / "splits" / "split_assignments.csv"

    # Code/config layout for this phase
    configs_dir: Path = PROJECT_ROOT / "configs"
    prompts_dir: Path = PROJECT_ROOT / "prompts" / "classification"
    results_dir: Path = PROJECT_ROOT / "results"
    models_dir: Path = PROJECT_ROOT / "models"

    # Canonical split
    random_seed: int = 42
    train_frac: float = 0.70
    dev_frac: float = 0.15
    test_frac: float = 0.15

    # Labels (fixed order used everywhere: confusion matrices, reports, etc.)
    labels: tuple[str, str] = ("not_sarcastic", "sarcastic")
    positive_label: str = "sarcastic"

    # API keys (reused pattern from config/settings.py; same .env file)
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY") or None
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or None


classification_settings = ClassificationSettings()
