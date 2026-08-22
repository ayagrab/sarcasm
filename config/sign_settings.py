"""Settings for the SIGN generalization phase (Part III of this project).

Kept separate from `config/classification_settings.py` (Part II, Dataset
A) so Part II's frozen paths/config are never accidentally reused or
overwritten by this new phase. See `PROJECT_SUMMARY.md`'s "Part III"
section for the full results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SignSettings:
    project_root: Path = PROJECT_ROOT

    # Raw SIGN files -- already present in the repo (staged by an earlier
    # part of the project), read-only, official SIGN train/dev/test split.
    # No header; 2 columns per row: (sarcastic_original, human_interpretation).
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    raw_files: tuple = (
        ("train", PROJECT_ROOT / "data" / "raw" / "original_train_dataset.csv"),
        ("dev", PROJECT_ROOT / "data" / "raw" / "original_dev_dataset.csv"),
        ("test", PROJECT_ROOT / "data" / "raw" / "original_test_dataset.csv"),
    )

    # Derived, durable artifacts (Phase 1 output) -- one long-format family
    # table per split, so downstream phases/sessions never need to
    # re-parse the raw files.
    processed_dir: Path = PROJECT_ROOT / "data" / "sign"

    # Code/config/results layout for this phase (distinct from Part II's
    # configs_dir/results_dir so the two studies never collide).
    configs_dir: Path = PROJECT_ROOT / "configs" / "sign"
    results_dir: Path = PROJECT_ROOT / "results" / "sign"
    train_variants_dir: Path = PROJECT_ROOT / "data" / "sign" / "train_variants"

    # Labels -- identical convention to Part II (config/classification_settings.py)
    # so SIGN examples are directly compatible with the existing
    # evaluation.metrics / evaluation.io helpers, no remapping needed.
    labels: tuple = ("not_sarcastic", "sarcastic")
    positive_label: str = "sarcastic"
    original_label: str = "sarcastic"
    interpretation_label: str = "not_sarcastic"

    # Deterministic sampling, matching Part II's convention
    # (config/classification_settings.py: random_seed = 42).
    random_seed: int = 42


sign_settings = SignSettings()
