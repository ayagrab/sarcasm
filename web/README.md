# Sarcasm Detector — web application

A demo web app for the sarcasm *detection* research project
(`src/classification/`, see the repo root's `EXPERIMENT_LOG.md` and
`PROJECT_SUMMARY.md`). Two modes:

- **Simple Mode** (`/`) — enter a sentence, get Sarcastic / Not Sarcastic
  and a confidence score, using whichever method is currently the
  configured *production model*.
- **Research Mode** (`/research`) — enter a sentence, see every
  implemented method's prediction side by side, with agreement/disagreement
  called out.
- **About** (`/about`) — plain-language explanation of each approach, plus
  (once Stage B legitimately finishes) the real final TEST metrics table.

This app is intentionally a thin presentation layer. It does not
re-implement any model/prompt/training logic — every prediction goes
through the exact same code (`src/classification/...`) used by the
experiment pipeline, via a small set of inference adapters (see
"Architecture" below). It never tunes a prompt, config, or hyperparameter
based on how a sentence typed into the UI comes out.

## Architecture

```text
Browser
   |
   v
Next.js frontend (web/frontend/, port 3000)
   |
   v  HTTP (fetch, NEXT_PUBLIC_API_BASE_URL)
FastAPI backend (web/backend/, port 8000)
   |
   +--> TfidfAdapter    -> src.classification.classical.tfidf_baseline
   +--> QwenAdapter x3  -> src.classification.llm.local_client / run_llm_classification
   +--> DspyAdapter     -> src.classification.dspy_pipeline.local_lm / signatures
   +--> DebertaAdapter  -> src.classification.transformer.finetune's checkpoint output
```

Each adapter (`web/backend/app/adapters/*.py`) implements one interface
(`BaseAdapter`: `status()` -> one of `AVAILABLE` / `NOT_TRAINED_YET` /
`NOT_FROZEN_YET` / `UNAVAILABLE`, and `predict(text)`). The API layer
(`app/main.py`) never special-cases a method by name; it iterates
`app/registry.py`'s registry and only ever calls `predict()` on an adapter
whose `status()` is `AVAILABLE`.

### The frozen-configuration gate

`app/frozen_registry.py` is the single switch controlling which methods
are allowed to serve real predictions. Per the project's TEST-sealing
methodology (`STAGE_B_CHECKLIST.md`, "PHASE 2 -- Freeze + sealed TEST
evaluation"), a method is only production-usable once Stage B has
selected and frozen exactly one configuration for it -- not merely once
its code happens to run without error. Concretely:

- `tfidf` (M1) is always considered frozen -- it was frozen back in Stage
  A, independently of the Stage B DEV/TEST workflow this file otherwise
  gates.
- Every other method reports `NOT_FROZEN_YET` (or `NOT_TRAINED_YET` for
  DeBERTa before it's even been trained) until
  `results/frozen_configs.json` exists and names it, e.g.:

  ```json
  {
    "production_model": "deberta",
    "frozen": {
      "tfidf": {"experiment_id": "EXP-001", "config_path": "configs/tfidf.json"},
      "deberta": {"experiment_id": "EXP-009", "config_path": "configs/transformer_deberta_v3_base.json"}
    }
  }
  ```

  This file is written once, by hand, at Stage B Phase 2 freeze time --
  not by this web app. Until it exists, `production_model` defaults to
  `tfidf`, documented in code as an interim baseline, not a scientific
  conclusion.

Loading a heavy model (Qwen, DeBERTa) is always lazy and only ever
attempted once a method's frozen check has already passed -- so this
backend starts and runs fine on a machine with no CUDA GPU and no trained
checkpoints (e.g. a laptop), it just reports everything but `tfidf` as not
yet available.

## Installation

Backend (reuses the main repo's Python environment -- the adapters import
`src.classification.*` and `config.classification_settings` directly, so
that environment needs `requirements.txt` + `requirements-classification.txt`
installed already; see the repo root README):

```bash
cd web/backend
pip install -r requirements.txt   # fastapi/uvicorn/httpx/pytest only
```

Frontend:

```bash
cd web/frontend
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_BASE_URL, defaults to localhost:8000
```

## Local development

Backend:

```bash
cd web/backend
uvicorn main:app --reload
```

Frontend (separate terminal):

```bash
cd web/frontend
npm run dev
```

Open http://localhost:3000.

## API endpoints

- `GET  /health` -- liveness check.
- `GET  /methods` -- every method's `{method, display_name, status, description}`.
- `GET  /methods/{method}` -- single method, 404 if unknown.
- `POST /predict` -- Simple Mode.
  ```json
  // request
  {"text": "Oh wonderful, another software update."}
  // response
  {"label": "sarcastic", "confidence": 0.93, "model": "deberta", "runtime_seconds": 0.12}
  ```
  Returns `503` if the configured production model isn't `AVAILABLE`.
- `POST /compare` -- Research Mode. Runs every registered method; each
  entry carries its own `status` and, only when `AVAILABLE`, a `label` /
  `confidence` (`null` for methods that don't produce a calibrated
  probability, e.g. a raw LLM chat completion -- never fabricated).
  ```json
  {
    "text": "Oh wonderful, another software update.",
    "predictions": [
      {"method": "tfidf", "display_name": "TF-IDF + Logistic Regression", "status": "AVAILABLE", "label": "sarcastic", "confidence": 0.78, "runtime_seconds": 0.003, "error": null},
      {"method": "qwen_zero_shot", "display_name": "Qwen Zero-shot", "status": "NOT_FROZEN_YET", "label": null, "confidence": null, "runtime_seconds": null, "error": null}
    ],
    "agreement": null
  }
  ```

Input validation (both endpoints): empty/whitespace-only text and text
over 2,000 characters are rejected with `422`. A method that raises during
`predict()` never fails the whole `/compare` request -- that one entry
gets `status: "UNAVAILABLE"` and an `error` message instead.

## Switching the production model

Edit (or create) `results/frozen_configs.json`'s `production_model` key --
no code change required. `app/registry.py:production_adapter()` reads it
on every `/predict` call (cheap: it's a small local JSON read, not a
network call).

## Where frozen artifacts are expected

- TF-IDF: refit in-process at backend startup from `data/splits/train.csv`
  using `configs/tfidf.json`'s frozen hyperparameters (cheap enough that
  persisting a pickle isn't worth the added state).
- Qwen (zero-shot/few-shot/reasoning) / DSPy: loaded from the Hugging Face
  cache (`HF_HOME`) the same way the experiment pipeline does --
  `Qwen/Qwen3-4B-Instruct-2507`. Requires a CUDA GPU; reports
  `UNAVAILABLE` (not a crash) if none is present.
- DeBERTa: `models/<experiment_id>/best_checkpoint/` (default
  `EXP-009`, or whatever `frozen_configs.json` names) -- the output of
  `src.classification.transformer.finetune`. `NOT_TRAINED_YET` if that
  directory doesn't exist at all.

## Testing

```bash
cd web/backend
pytest
```

All tests pass without a GPU, without `dspy` needing real inference, and
without a trained DeBERTa checkpoint -- heavy model loading is mocked in
`tests/test_adapters.py`; `tests/test_api.py` exercises the real TF-IDF
path (fast, CPU-only) plus every validation/error-handling branch.

## Deployment considerations

- The backend is a plain ASGI app (`app.main:app`) -- deployable behind
  any ASGI server/reverse proxy. `CORSMiddleware`'s `allow_origins` is
  currently hardcoded to `http://localhost:3000` for local dev; update it
  (or read it from an env var) before deploying the frontend anywhere
  else.
- Do not assume the deployment machine has the same 2x Tesla M60 GPUs as
  the Stage B experiment VM. The frozen-configuration gate already makes
  Qwen/DSPy availability purely a function of whether a CUDA GPU + the
  cached model are present -- if not, they degrade to `UNAVAILABLE`
  rather than crashing the app; Simple Mode (TF-IDF today, DeBERTa once
  frozen -- both CPU-friendly) keeps working regardless.
- Model loading is lazy and cached per-process (module-level singletons in
  `app/registry.py` and each adapter) -- never reloaded per request; a
  restart is required to pick up a newly-written `frozen_configs.json` or
  a newly-trained checkpoint.
