"""LLM client factory for the classification pipeline, configurable by provider.

Two providers are wired up:
- "openrouter" (default): reuses the existing, already tested
  `src.common.openrouter_client`, matching this repository's
  OpenRouter-first convention for the interpretation pipeline.
- "local_hf": a local Hugging Face Transformers model on GPU (e.g. Qwen3-4B
  on a CUDA-capable machine) -- see `src.classification.llm.local_client`,
  which raises immediately if no CUDA GPU is visible.
"""
from __future__ import annotations

from src.common.openrouter_client import get_openrouter_client

SUPPORTED_PROVIDERS = ("openrouter", "local_hf")


def get_llm_client(provider: str = "openrouter", **kwargs):
    if provider == "openrouter":
        return get_openrouter_client()
    if provider == "local_hf":
        from src.classification.llm.local_client import get_local_hf_client

        return get_local_hf_client(**kwargs)
    raise ValueError(f"Unsupported provider: {provider!r} (supported: {SUPPORTED_PROVIDERS})")
