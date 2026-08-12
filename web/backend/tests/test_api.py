"""API-level tests. TF-IDF is real (fast, CPU, no external dependency --
fits in-process at adapter construction) so its /predict path is tested
for real. Everything else is expected to report a non-AVAILABLE status in
this environment (no Stage B freeze yet, no GPU/checkpoint on a plain dev
machine) -- these tests assert that honestly, and never require loading
Qwen/DeBERTa to pass."""
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


def test_non_frozen_methods_report_honest_status():
    response = client.get("/methods")
    by_method = {m["method"]: m["status"] for m in response.json()}
    assert by_method["qwen_zero_shot"] == ModelStatus.NOT_FROZEN_YET.value
    assert by_method["qwen_few_shot"] == ModelStatus.NOT_FROZEN_YET.value
    assert by_method["qwen_reasoning"] == ModelStatus.NOT_FROZEN_YET.value
    assert by_method["dspy"] == ModelStatus.NOT_FROZEN_YET.value
    # deberta: NOT_TRAINED_YET takes priority when no checkpoint exists at all
    assert by_method["deberta"] in (ModelStatus.NOT_TRAINED_YET.value, ModelStatus.NOT_FROZEN_YET.value)


def test_get_unknown_method_404():
    response = client.get("/methods/not_a_real_method")
    assert response.status_code == 404


def test_predict_valid_sentence():
    response = client.post("/predict", json={"text": "Oh wonderful, another meeting that could have been an email."})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("sarcastic", "not_sarcastic")
    assert body["model"] == "tfidf"  # default production_model, no frozen_configs.json present in tests
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
    # A not-frozen-yet method must carry no fabricated prediction.
    qwen_entry = next(p for p in body["predictions"] if p["method"] == "qwen_zero_shot")
    assert qwen_entry["status"] == ModelStatus.NOT_FROZEN_YET.value
    assert qwen_entry["label"] is None
    assert qwen_entry["confidence"] is None


def test_compare_empty_text_rejected():
    response = client.post("/compare", json={"text": ""})
    assert response.status_code == 422
