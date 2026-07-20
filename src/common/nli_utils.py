"""Label-mapping helpers for NLI-based evaluation.

Extracted out of `src/evaluation/evaluate_with_nli.py` so the
entailment-vs-contradiction decision logic can be unit tested with mocked
probabilities/labels, without downloading any model
(see tests/test_nli_utils.py).
"""
from __future__ import annotations
from typing import Mapping, Optional

# Fallback mapping for MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (this
# project's default NLI model, config.models.JUDGE_MODELS["nli"]), as
# documented on its model card. Used only if a model's own
# `config.id2label` does not expose recognizable label names (e.g. generic
# "LABEL_0"/"LABEL_1"/"LABEL_2" placeholders) -- numeric label order is
# never assumed to be universal across NLI models.
DEFAULT_MODEL_LABEL_FALLBACK: Mapping[int, str] = {0: "entailment", 1: "neutral", 2: "contradiction"}

_RECOGNIZED_LABELS = {"entailment", "neutral", "contradiction"}


def normalize_id2label(id2label: Mapping[int, str]) -> dict[str, int]:
    """Map lowercase label name -> index, tolerating case differences."""
    return {str(label).strip().lower(): int(idx) for idx, label in id2label.items()}


def resolve_label_index(
    id2label: Mapping[int, str], label: str, fallback: Optional[Mapping[int, str]] = None
) -> Optional[int]:
    """Find the index of `label` (e.g. "entailment") in id2label, case-insensitively.

    If none of `id2label`'s values are recognizable NLI label names at all
    (e.g. generic "LABEL_0"/"LABEL_1" placeholders), falls back to the
    explicit `fallback` mapping. Returns None if `label` cannot be resolved
    either way.
    """
    normalized = normalize_id2label(id2label)
    if label in normalized:
        return normalized[label]

    has_recognizable_labels = any(str(name).strip().lower() in _RECOGNIZED_LABELS for name in id2label.values())
    if not has_recognizable_labels and fallback is not None:
        return normalize_id2label(fallback).get(label)
    return None


def classify_entailment(
    probabilities: list[float],
    id2label: Mapping[int, str],
    fallback: Optional[Mapping[int, str]] = DEFAULT_MODEL_LABEL_FALLBACK,
) -> int:
    """Return 1 if entailment probability > contradiction probability, else 0.

    `probabilities` must be aligned with `id2label`'s indices (e.g. the
    softmax output of an NLI model's logits, for a single example).

    Raises ValueError if the entailment/contradiction labels cannot be
    resolved from `id2label` (or `fallback`) at all -- this project never
    assumes a fixed numeric label order.
    """
    entailment_idx = resolve_label_index(id2label, "entailment", fallback)
    contradiction_idx = resolve_label_index(id2label, "contradiction", fallback)
    if entailment_idx is None or contradiction_idx is None:
        raise ValueError(
            f"Could not resolve 'entailment'/'contradiction' labels from id2label={id2label!r}. "
            "Pass an explicit `fallback` mapping appropriate for this model."
        )
    return 1 if probabilities[entailment_idx] > probabilities[contradiction_idx] else 0
