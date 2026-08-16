"""Adapter-level unit tests, isolated from the API layer. Heavy model
loading (Qwen/DSPy) is mocked -- these tests must never require a GPU or
a real model download to pass."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app import PROJECT_ROOT
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


@patch("app.adapters.qwen_adapter.is_frozen", return_value=True)
@patch("app.adapters.qwen_adapter.frozen_config_path")
@patch("src.classification.llm.local_client.get_local_hf_client")
@patch("src.classification.llm.few_shot_selection.select_random_few_shot")
@patch("src.classification.llm.few_shot_selection.select_curated_few_shot")
def test_qwen_few_shot_uses_registry_config_not_hardcoded_curated(
    mock_curated, mock_random, _mock_client, mock_config_path, _mock_frozen, tmp_path
):
    import pandas as pd

    # The registry names the frozen RANDOM-8 config (Phase 2's actual
    # freeze decision) -- the adapter must follow it, not fall back to a
    # hardcoded curated config regardless of what's actually frozen.
    fake_config = tmp_path / "frozen_few_shot.json"
    fake_config.write_text(json.dumps({"few_shot_variant": "random", "n_shots": 8, "seed": 42}))
    mock_config_path.return_value = str(fake_config)
    mock_random.return_value = pd.DataFrame({"text": ["demo"], "label": ["sarcastic"]})

    adapter = QwenAdapter("qwen_few_shot")
    adapter._lazy_load()

    mock_random.assert_called_once()
    mock_curated.assert_not_called()


@patch("app.adapters.dspy_adapter.is_frozen", return_value=False)
def test_dspy_adapter_not_frozen_never_loads_model(_mock_frozen):
    adapter = DspyAdapter()
    assert adapter.status() == ModelStatus.NOT_FROZEN_YET
    assert adapter._program is None


@patch("app.adapters.dspy_adapter.is_frozen", return_value=True)
@patch("app.adapters.dspy_adapter.frozen_experiment_id", return_value="EXP-008")
@patch("src.classification.dspy_pipeline.local_lm.LocalQwenLM")
def test_dspy_adapter_loads_frozen_compiled_program(_mock_lm, _mock_exp_id, _mock_frozen):
    import dspy

    # The real frozen MIPROv2 program, already on disk from Stage B Phase 2.
    compiled_path = PROJECT_ROOT / "results" / "EXP-008" / "compiled_program.json"
    assert compiled_path.exists(), "fixture assumption: EXP-008's compiled program must exist"
    expected_instructions = json.loads(compiled_path.read_text())["signature"]["instructions"]

    with patch("dspy.configure"):
        adapter = DspyAdapter()
        adapter._lazy_load()

    assert adapter._init_error is None
    assert isinstance(adapter._program, dspy.Predict)
    assert adapter._program.signature.instructions == expected_instructions


@patch("app.adapters.dspy_adapter.is_frozen", return_value=True)
@patch("app.adapters.dspy_adapter.frozen_experiment_id", return_value="NOT-A-REAL-EXPERIMENT")
@patch("src.classification.dspy_pipeline.local_lm.LocalQwenLM")
def test_dspy_adapter_falls_back_to_unoptimized_when_compiled_program_missing(_mock_lm, _mock_exp_id, _mock_frozen):
    with patch("dspy.configure"):
        adapter = DspyAdapter()
        adapter._lazy_load()

    # No crash, no fabricated program -- just the plain unoptimized baseline.
    assert adapter._init_error is None
    assert adapter._program is not None


@patch("app.adapters.deberta_adapter.DebertaAdapter._checkpoint_dir")
def test_deberta_adapter_no_checkpoint_reports_not_trained_yet(mock_dir, tmp_path):
    mock_dir.return_value = tmp_path / "does_not_exist"
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
