"""API-level tests. TF-IDF is real (fast, CPU, no external dependency --
fits in-process at adapter construction) so its /predict path is tested
for real. Every method is FROZEN as of Stage B Phase 2
(`results/frozen_configs.json` exists), but Qwen/DSPy still report
UNAVAILABLE on a plain CPU-only dev machine with no CUDA GPU -- these
tests assert that honestly, and never require loading Qwen to pass.
DeBERTa's status is left unasserted precisely (it depends on whether this
machine's `transformers` version can load the checkpoint's tokenizer --
see STAGE_B_CHECKLIST.md) rather than pinned to a snapshot of one
machine's current state."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ModelStatus

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_methods_lists_all_six():
    response = client.get("/methods")
    assert response.status_code == 200
    methods = {m["method"] for m in response.json()}
    assert methods == {"tfidf", "qwen_zero_shot", "qwen_few_shot", "qwen_reasoning", "dspy", "deberta"}


def test_tfidf_is_frozen_and_available():
    response = client.get("/methods/tfidf")
    assert response.status_code == 200
    assert response.json()["status"] == ModelStatus.AVAILABLE.value


def test_frozen_but_gpu_only_methods_report_honest_status():
    response = client.get("/methods")
    by_method = {m["method"]: m["status"] for m in response.json()}
    # Frozen since Phase 2, but this test machine has no CUDA GPU -- must
    # never fabricate AVAILABLE, and must never crash instead of reporting.
    assert by_method["qwen_zero_shot"] == ModelStatus.UNAVAILABLE.value
    assert by_method["qwen_few_shot"] == ModelStatus.UNAVAILABLE.value
    assert by_method["qwen_reasoning"] == ModelStatus.UNAVAILABLE.value
    assert by_method["dspy"] == ModelStatus.UNAVAILABLE.value
    # deberta doesn't need a GPU, so its status here depends on whether
    # this machine's transformers version can load the checkpoint -- both
    # outcomes are honest, neither is a bug in the adapter itself.
    assert by_method["deberta"] in (ModelStatus.AVAILABLE.value, ModelStatus.UNAVAILABLE.value)


def test_get_unknown_method_404():
    response = client.get("/methods/not_a_real_method")
    assert response.status_code == 404


def test_predict_valid_sentence():
    from app.registry import production_adapter

    response = client.post("/predict", json={"text": "Oh wonderful, another meeting that could have been an email."})
    if production_adapter().status() != ModelStatus.AVAILABLE:
        # production_model (deberta, per frozen_configs.json) isn't loadable
        # on this machine -- must be an honest 503, never a fabricated prediction.
        assert response.status_code == 503
        return
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("sarcastic", "not_sarcastic")
    assert body["model"] == "deberta"  # production_model per results/frozen_configs.json
    assert isinstance(body["runtime_seconds"], float)


def test_predict_empty_text_rejected():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_predict_too_long_rejected():
    response = client.post("/predict", json={"text": "a" * 3000})
    assert response.status_code == 422


def test_predict_malformed_request_rejected():
    response = client.post("/predict", json={"not_text": "oops"})
    assert response.status_code == 422


def test_compare_returns_all_methods():
    response = client.post("/compare", json={"text": "Thank you for helping me with the assignment."})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Thank you for helping me with the assignment."
    assert len(body["predictions"]) == 6
    tfidf_entry = next(p for p in body["predictions"] if p["method"] == "tfidf")
    assert tfidf_entry["status"] == ModelStatus.AVAILABLE.value
    assert tfidf_entry["label"] in ("sarcastic", "not_sarcastic")
    # A method that's frozen but can't actually run here (no GPU) must
    # carry no fabricated prediction.
    qwen_entry = next(p for p in body["predictions"] if p["method"] == "qwen_zero_shot")
    assert qwen_entry["status"] == ModelStatus.UNAVAILABLE.value
    assert qwen_entry["label"] is None
    assert qwen_entry["confidence"] is None


def test_compare_empty_text_rejected():
    response = client.post("/compare", json={"text": ""})
    assert response.status_code == 422
