"""Tests for src/classification/llm/local_client.py's safety guards --
the parts that don't require an actual GPU. No model is downloaded or
loaded here; these only check that unsupported-on-Maxwell configurations
(bfloat16, FlashAttention2) and a missing CUDA device are rejected before
any download/load would happen."""
from __future__ import annotations

import pytest

from src.classification.llm.local_client import LocalHFClient
from src.classification.llm import client as client_module


def test_rejects_bfloat16():
    with pytest.raises(ValueError, match="BF16"):
        LocalHFClient(dtype="bfloat16")


def test_rejects_flash_attention_2():
    with pytest.raises(ValueError, match="FlashAttention2"):
        LocalHFClient(attn_implementation="flash_attention_2")


def test_rejects_missing_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="No CUDA GPU"):
        LocalHFClient(dtype="float16", attn_implementation="eager")


def test_get_llm_client_supports_local_hf_provider_name():
    assert "local_hf" in client_module.SUPPORTED_PROVIDERS


def test_get_llm_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        client_module.get_llm_client("azure_openai")
