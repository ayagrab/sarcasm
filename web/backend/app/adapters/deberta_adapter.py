"""Fine-tuned DeBERTa-v3-base adapter.

Loads the checkpoint written by `src.classification.transformer.finetune`
(`models/<experiment_id>/best_checkpoint/`) once, lazily, the first time
it's actually needed -- small enough (~370MB) to run comfortably on CPU
or MPS, so unlike the Qwen adapters this one is expected to work on the
local dev machine too, not just the Azure GPU VM.

Status precedence: NOT_TRAINED_YET (no checkpoint on disk at all) takes
priority over NOT_FROZEN_YET (checkpoint exists, from Stage B DEV
development, but Phase 2 hasn't frozen it as the final M6 configuration)
-- these are deliberately different signals; conflating them would hide
whether the *blocker* is "hasn't been trained" vs. "hasn't been decided."
"""
from __future__ import annotations

from app import PROJECT_ROOT
from app.adapters.base import BaseAdapter
from app.frozen_registry import frozen_experiment_id, is_frozen
from app.schemas import ModelStatus

METHOD = "deberta"
DEFAULT_EXPERIMENT_ID = "EXP-009"
LABEL_ORDER = ["not_sarcastic", "sarcastic"]  # ID2LABEL in finetune.py: 0 -> not_sarcastic, 1 -> sarcastic


class DebertaAdapter(BaseAdapter):
    method = METHOD
    display_name = "Fine-tuned DeBERTa-v3-base"
    description = (
        "A pretrained English transformer encoder (microsoft/deberta-v3-base), fine-tuned "
        "end-to-end on the TRAIN split for this exact task -- unlike the LLM approaches, it "
        "has learned task-specific weights rather than being prompted."
    )

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._init_error: str | None = None
        self._loaded = False

    def _checkpoint_dir(self):
        experiment_id = frozen_experiment_id(self.method) or DEFAULT_EXPERIMENT_ID
        return PROJECT_ROOT / "models" / experiment_id / "best_checkpoint"

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        checkpoint_dir = self._checkpoint_dir()
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=False)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(checkpoint_dir), use_safetensors=True
            )
            self._model.eval()
        except Exception as exc:
            self._init_error = str(exc)

    def status(self) -> ModelStatus:
        if not self._checkpoint_dir().exists():
            return ModelStatus.NOT_TRAINED_YET
        if not is_frozen(self.method):
            return ModelStatus.NOT_FROZEN_YET
        self._lazy_load()
        if self._model is None:
            return ModelStatus.UNAVAILABLE
        return ModelStatus.AVAILABLE

    def _predict_raw(self, text: str) -> tuple[str, float | None]:
        import torch

        inputs = self._tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(probs).item())
        return LABEL_ORDER[pred_id], float(probs[pred_id].item())
