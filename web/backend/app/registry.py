"""Central adapter registry -- the only place that lists all six methods.
`main.py` never special-cases a method by name; it iterates this registry.
"""
from __future__ import annotations

from app.adapters.base import BaseAdapter
from app.adapters.deberta_adapter import DebertaAdapter
from app.adapters.dspy_adapter import DspyAdapter
from app.adapters.qwen_adapter import QwenAdapter
from app.adapters.tfidf_adapter import TfidfAdapter
from app.frozen_registry import get_production_model

_REGISTRY: dict[str, BaseAdapter] | None = None


def _build_registry() -> dict[str, BaseAdapter]:
    return {
        "tfidf": TfidfAdapter(),
        "qwen_zero_shot": QwenAdapter("qwen_zero_shot"),
        "qwen_few_shot": QwenAdapter("qwen_few_shot"),
        "qwen_reasoning": QwenAdapter("qwen_reasoning"),
        "dspy": DspyAdapter(),
        "deberta": DebertaAdapter(),
    }


def get_registry() -> dict[str, BaseAdapter]:
    """Built once per process (module-level singleton), so heavy adapters
    (TF-IDF's fit; any later-loaded GPU model) aren't reconstructed per
    request. Adapters that need real model loading do it lazily inside
    their own `status()`/`predict()`, not here -- constructing this dict
    must stay cheap and never raise."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_adapter(method: str) -> BaseAdapter | None:
    return get_registry().get(method)


def production_adapter() -> BaseAdapter:
    method = get_production_model()
    adapter = get_adapter(method)
    if adapter is None:
        raise RuntimeError(f"Configured production_model {method!r} has no matching adapter")
    return adapter
