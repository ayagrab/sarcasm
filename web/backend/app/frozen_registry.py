"""Tracks which methods have a Stage B FROZEN configuration.

This is the single switch the rest of the web app checks before ever
calling a non-M1 method for real -- it is deliberately NOT hardcoded
"true" for anything still in Stage B development. Per the project's
methodology (see EXPERIMENT_LOG.md / STAGE_B_CHECKLIST.md "PHASE 2 --
Freeze + sealed TEST evaluation"), a method only becomes production-usable
once Stage B has selected and frozen exactly one configuration for it --
not merely once it happens to run without error.

`results/frozen_configs.json` is written once, at Phase 2 freeze time, by
the experiment pipeline (not by this web app) -- see
STAGE_B_CHECKLIST.md's Phase 2 checklist. Until that file exists, every
method except `tfidf` (already frozen back in Stage A, independently of
Stage B) reports NOT_FROZEN_YET/NOT_TRAINED_YET regardless of whether its
underlying inference code happens to run.

Expected shape once written:
    {
      "production_model": "deberta",
      "frozen": {
        "tfidf": {"experiment_id": "EXP-001", "config_path": "configs/tfidf.json"},
        "qwen_zero_shot": {"experiment_id": "EXP-002", "config_path": "configs/llm_zero_shot_qwen_local.json"},
        ...
      }
    }
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FROZEN_REGISTRY_PATH = PROJECT_ROOT / "results" / "frozen_configs.json"

# M1 (TF-IDF + Logistic Regression) was frozen in Stage A, independently
# of the Stage B DEV/TEST-sealing workflow this registry file tracks --
# see EXPERIMENT_LOG.md, EXP-001. Hardcoded here rather than requiring
# the (not-yet-existent) Stage B freeze file just to unlock the one
# method that was never part of Stage B's freeze process.
_ALWAYS_FROZEN = {"tfidf"}


def _load_registry() -> dict:
    if not FROZEN_REGISTRY_PATH.exists():
        return {"production_model": None, "frozen": {}}
    with open(FROZEN_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def is_frozen(method: str) -> bool:
    if method in _ALWAYS_FROZEN:
        return True
    registry = _load_registry()
    return method in registry.get("frozen", {})


def frozen_experiment_id(method: str) -> str | None:
    registry = _load_registry()
    entry = registry.get("frozen", {}).get(method)
    return entry.get("experiment_id") if entry else None


def frozen_config_path(method: str) -> str | None:
    """The exact config file Stage B Phase 2 froze for this method (e.g.
    which few-shot variant -- random vs. curated -- won), so adapters
    never have to hardcode a guess that can silently drift out of sync
    with the actual freeze decision."""
    registry = _load_registry()
    entry = registry.get("frozen", {}).get(method)
    return entry.get("config_path") if entry else None


def get_production_model() -> str:
    """Which method backs Simple Mode. Defaults to `tfidf` -- the only
    method frozen so far -- and is meant to be swapped the moment Stage B
    legitimately freezes a stronger method (edit `production_model` in
    `results/frozen_configs.json`, no code change required)."""
    registry = _load_registry()
    return registry.get("production_model") or "tfidf"
