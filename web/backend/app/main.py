"""FastAPI app: sarcasm classification demo backend.

    POST /predict   -- Simple Mode. Uses whichever method is currently
                        configured as `production_model` in
                        results/frozen_configs.json (defaults to `tfidf`,
                        the only method frozen before Stage B started).
    POST /compare    -- Research Mode. Runs every registered method and
                        returns one entry per method, each carrying its own
                        honest status -- never a fabricated prediction for
                        a method that isn't ready.
    GET  /methods    -- Lists every method + its current status/description,
                        for the frontend's "about the approaches" section.
    GET  /health     -- Liveness check.

Run:
    cd web/backend && uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.registry import get_adapter, get_registry, production_adapter
from app.schemas import CompareRequest, CompareResponse, MethodInfo, MethodPrediction, ModelStatus, PredictRequest, PredictResponse

logger = logging.getLogger("sarcasm_web")

app = FastAPI(title="Sarcasm Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/methods", response_model=list[MethodInfo])
def list_methods() -> list[MethodInfo]:
    return [
        MethodInfo(
            method=method,
            display_name=adapter.display_name,
            status=adapter.status(),
            description=adapter.description,
        )
        for method, adapter in get_registry().items()
    ]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    adapter = production_adapter()
    status = adapter.status()
    if status != ModelStatus.AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Production model ({adapter.method}) is currently {status.value}, not available.",
        )
    try:
        result = adapter.predict(request.text)
    except Exception:
        logger.exception("predict() failed for method=%s", adapter.method)
        raise HTTPException(status_code=502, detail=f"{adapter.display_name} failed to produce a prediction.")
    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        model=adapter.method,
        runtime_seconds=round(result.runtime_seconds, 4),
    )


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    predictions: list[MethodPrediction] = []
    available_labels: set[str] = set()

    for method, adapter in get_registry().items():
        status = adapter.status()
        if status != ModelStatus.AVAILABLE:
            predictions.append(
                MethodPrediction(method=method, display_name=adapter.display_name, status=status)
            )
            continue
        try:
            result = adapter.predict(request.text)
        except Exception as exc:
            logger.exception("compare(): predict() failed for method=%s", method)
            predictions.append(
                MethodPrediction(
                    method=method,
                    display_name=adapter.display_name,
                    status=ModelStatus.UNAVAILABLE,
                    error=str(exc),
                )
            )
            continue
        available_labels.add(result.label)
        predictions.append(
            MethodPrediction(
                method=method,
                display_name=adapter.display_name,
                status=status,
                label=result.label,
                confidence=result.confidence,
                runtime_seconds=round(result.runtime_seconds, 4),
            )
        )

    agreement = len(available_labels) == 1 if len(available_labels) >= 1 else None
    return CompareResponse(text=request.text, predictions=predictions, agreement=agreement)


def _method_not_found(method: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Unknown method: {method}")


@app.get("/methods/{method}", response_model=MethodInfo)
def get_method(method: str) -> MethodInfo:
    adapter = get_adapter(method)
    if adapter is None:
        raise _method_not_found(method)
    return MethodInfo(
        method=method, display_name=adapter.display_name, status=adapter.status(), description=adapter.description
    )
