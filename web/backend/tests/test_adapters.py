"""Adapter-level unit tests, isolated from the API layer. Heavy model
loading (Qwen/DSPy) is mocked -- these tests must never require a GPU or
a real model download to pass."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.adapters.deberta_adapter import DebertaAdapter
from app.adapters.dspy_adapter import DspyAdapter
from app.adapters.qwen_adapter import QwenAdapter
from app.adapters.tfidf_adapter import TfidfAdapter
from app.schemas import ModelStatus


def test_tfidf_adapter_fits_and_predicts():
    adapter = TfidfAdapter()
    assert adapter.status() == ModelStatus.AVAILABLE
    result = adapter.predict("Oh great, another Monday.")
    assert result.label in ("sarcastic", "not_sarcastic")
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0


@patch("app.adapters.qwen_adapter.is_frozen", return_value=False)
def test_qwen_adapter_not_frozen_never_loads_model(_mock_frozen):
    adapter = QwenAdapter("qwen_zero_shot")
    assert adapter.status() == ModelStatus.NOT_FROZEN_YET
    # The heavy client must never have been constructed pre-freeze.
    assert adapter._client is None
    assert adapter._loaded is False


@patch("app.adapters.qwen_adapter.is_frozen", return_value=True)
@patch("src.classification.llm.local_client.get_local_hf_client", side_effect=RuntimeError("No CUDA GPU visible"))
def test_qwen_adapter_frozen_but_no_gpu_reports_unavailable(_mock_client, _mock_frozen):
    adapter = QwenAdapter("qwen_zero_shot")
    assert adapter.status() == ModelStatus.UNAVAILABLE
    assert "No CUDA GPU" in adapter._init_error


@patch("app.adapters.dspy_adapter.is_frozen", return_value=False)
def test_dspy_adapter_not_frozen_never_loads_model(_mock_frozen):
    adapter = DspyAdapter()
    assert adapter.status() == ModelStatus.NOT_FROZEN_YET
    assert adapter._program is None


def test_deberta_adapter_no_checkpoint_reports_not_trained_yet():
    adapter = DebertaAdapter()
    assert adapter.status() == ModelStatus.NOT_TRAINED_YET


@patch("app.adapters.deberta_adapter.is_frozen", return_value=True)
@patch("app.adapters.deberta_adapter.DebertaAdapter._checkpoint_dir")
def test_deberta_adapter_predicts_when_checkpoint_present(mock_dir, _mock_frozen, tmp_path):
    import torch

    mock_dir.return_value = tmp_path  # .exists() is True for any real tmp_path

    fake_tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
    fake_model = MagicMock(return_value=MagicMock(logits=torch.tensor([[0.1, 4.0]])))

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer), patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained", return_value=fake_model
    ):
        adapter = DebertaAdapter()
        assert adapter.status() == ModelStatus.AVAILABLE
        result = adapter.predict("Oh sure, that's exactly what I wanted.")
        assert result.label == "sarcastic"
        assert result.confidence > 0.9
