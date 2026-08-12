"""Common adapter interface every method (TF-IDF, Qwen x3 modes, DSPy,
DeBERTa) implements, so the API layer (`main.py`) never special-cases a
method by name -- it only ever calls `status()` / `predict()`.

Adapters must never crash the process on init failure (e.g. no CUDA GPU
for the local Qwen client) -- failures are caught and turned into an
UNAVAILABLE status, checked by the caller *before* `predict()` is ever
called. `predict()` itself is only ever invoked when `status()` returned
AVAILABLE, so it may assume its model/pipeline is ready.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas import ModelStatus


@dataclass
class PredictionResult:
    label: str
    confidence: float | None
    runtime_seconds: float


class BaseAdapter(ABC):
    method: str
    display_name: str
    description: str

    @abstractmethod
    def status(self) -> ModelStatus: ...

    @abstractmethod
    def _predict_raw(self, text: str) -> tuple[str, float | None]:
        """Returns (label, confidence). `confidence` is None for methods
        that don't naturally produce a calibrated probability (e.g. a
        plain LLM chat completion with no logprobs) -- never fabricate one."""

    def predict(self, text: str) -> PredictionResult:
        start = time.monotonic()
        label, confidence = self._predict_raw(text)
        runtime = time.monotonic() - start
        return PredictionResult(label=label, confidence=confidence, runtime_seconds=runtime)
