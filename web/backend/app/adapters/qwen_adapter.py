"""Local Qwen3-4B-Instruct-2507 adapter (zero-shot / few-shot / reasoning).

Reuses `classify_one`/`build_messages` from the existing manual-prompt LLM
pipeline directly -- the web app never re-implements prompting/parsing
logic, and never constructs a prompt "because it looks better in the
demo" (see `prompts/classification/*.txt`, unchanged by this file).

Gated by `frozen_registry.is_frozen(method)`: until Stage B Phase 2
freezes a configuration for a given mode, `status()` reports
NOT_FROZEN_YET and the (expensive, GPU-only) local model is never even
loaded -- so this adapter is safe to construct on any machine, including
one with no CUDA GPU (the target deploy machine may not be the same 2x
Tesla M60 Azure VM used for experiments).
"""
from __future__ import annotations

import json

from app import PROJECT_ROOT
from app.adapters.base import BaseAdapter
from app.frozen_registry import frozen_config_path, is_frozen
from app.schemas import ModelStatus

CHECKPOINT = "Qwen/Qwen3-4B-Instruct-2507"

# Fallback config paths (used only if a method is frozen but the registry
# entry has no `config_path` -- shouldn't happen once Phase 2 has written
# `results/frozen_configs.json` for real, but keeps this adapter safe to
# construct even against a hand-edited/partial registry file) -- kept in
# sync with the Stage B config directory, never hand-edited "to make the
# demo look better".
DEFAULT_CONFIG_PATHS = {
    "qwen_zero_shot": PROJECT_ROOT / "configs" / "llm_zero_shot_qwen_local.json",
    "qwen_few_shot": PROJECT_ROOT / "configs" / "llm_few_shot_random_8_qwen_local.json",
    "qwen_reasoning": PROJECT_ROOT / "configs" / "llm_reasoning_qwen_local.json",
}
MODE_BY_METHOD = {
    "qwen_zero_shot": "zero_shot",
    "qwen_few_shot": "few_shot",
    "qwen_reasoning": "reasoning",
}
DISPLAY_NAMES = {
    "qwen_zero_shot": "Qwen Zero-shot",
    "qwen_few_shot": "Qwen Few-shot",
    "qwen_reasoning": "Qwen Structured Reasoning",
}
DESCRIPTIONS = {
    "qwen_zero_shot": (
        "A 4B-parameter instruction-tuned LLM is asked to classify the sentence directly, "
        "given only the task definition -- no examples, no chain of reasoning."
    ),
    "qwen_few_shot": (
        "The same LLM is shown a small set of labeled example sentences before classifying "
        "the new one, to see whether in-context examples improve on zero-shot."
    ),
    "qwen_reasoning": (
        "The same LLM is prompted to reason step-by-step about tone/context before committing "
        "to a label, to see whether explicit reasoning improves accuracy over direct prompting."
    ),
}


class QwenAdapter(BaseAdapter):
    def __init__(self, method: str) -> None:
        if method not in MODE_BY_METHOD:
            raise ValueError(f"Unknown Qwen method: {method}")
        self.method = method
        self.display_name = DISPLAY_NAMES[method]
        self.description = DESCRIPTIONS[method]
        self._mode = MODE_BY_METHOD[method]
        self._client = None
        self._init_error: str | None = None
        self._demonstrations: str | None = None
        self._loaded = False

    def _lazy_load(self) -> None:
        """Only ever called once `status()` has confirmed this method is
        frozen -- loading a multi-GB model is too expensive to attempt
        speculatively on every adapter construction."""
        if self._loaded:
            return
        self._loaded = True
        try:
            from src.classification.llm.local_client import get_local_hf_client

            self._client = get_local_hf_client(checkpoint=CHECKPOINT)
            if self._mode == "few_shot":
                self._load_demonstrations()
        except Exception as exc:  # no CUDA GPU on this machine, model not cached, etc.
            self._init_error = str(exc)

    def _load_demonstrations(self) -> None:
        import pandas as pd

        from config.classification_settings import classification_settings
        from src.classification.llm.few_shot_selection import select_curated_few_shot, select_random_few_shot
        from src.classification.llm.run_llm_classification import format_demonstrations

        registry_path = frozen_config_path(self.method)
        config_path = (PROJECT_ROOT / registry_path) if registry_path else DEFAULT_CONFIG_PATHS[self.method]
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        train_df = pd.read_csv(classification_settings.splits_dir / "train.csv")
        selector = select_random_few_shot if config.get("few_shot_variant", "curated") == "random" else select_curated_few_shot
        demo_df = selector(train_df, config.get("n_shots", 8), config.get("seed", 42))
        self._demonstrations = format_demonstrations(demo_df)

    def status(self) -> ModelStatus:
        if not is_frozen(self.method):
            return ModelStatus.NOT_FROZEN_YET
        self._lazy_load()
        if self._client is None:
            return ModelStatus.UNAVAILABLE
        return ModelStatus.AVAILABLE

    def _predict_raw(self, text: str) -> tuple[str, float | None]:
        from src.classification.llm.run_llm_classification import classify_one

        result = classify_one(
            self._client,
            CHECKPOINT,
            self._mode,
            text,
            temperature=0.0,
            demonstrations=self._demonstrations,
            use_cache=False,
        )
        # A raw chat completion has no calibrated probability attached --
        # report label only, never a fabricated confidence.
        return result["label"], None
