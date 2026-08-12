"""Pydantic request/response models for the web API. Kept separate from
the classification package's own dataclasses (`config/classification_settings.py`,
`evaluation/metrics.py`) -- this is a presentation-layer schema, not the
experiment-artifact schema those serve.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_LENGTH = 2000


class ModelStatus(str, Enum):
    """Honest availability signal for a method -- never fabricate a
    prediction for a method that isn't legitimately ready.

    AVAILABLE       -- frozen configuration + inference path working now.
    NOT_TRAINED_YET -- no checkpoint/trained artifact exists yet (e.g. M6
                        before EXP-009 has run).
    NOT_FROZEN_YET  -- inference is technically possible, but Stage B has
                        not frozen a final configuration for this method
                        yet (Phase 2 not reached) -- per the project's
                        TEST-sealing/no-premature-production-choice rule.
    UNAVAILABLE     -- inference path exists but failed to initialize on
                        this machine (e.g. no CUDA GPU for the local Qwen
                        client) -- not a data/config problem, an
                        environment one.
    """

    AVAILABLE = "AVAILABLE"
    NOT_TRAINED_YET = "NOT_TRAINED_YET"
    NOT_FROZEN_YET = "NOT_FROZEN_YET"
    UNAVAILABLE = "UNAVAILABLE"


class PredictRequest(BaseModel):
    text: str = Field(..., description="English sentence to classify.")

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be empty or whitespace-only")
        if len(stripped) > MAX_TEXT_LENGTH:
            raise ValueError(f"text exceeds the {MAX_TEXT_LENGTH}-character limit")
        return stripped


class PredictResponse(BaseModel):
    label: str
    confidence: float | None = None
    model: str
    runtime_seconds: float


class CompareRequest(PredictRequest):
    pass


class MethodPrediction(BaseModel):
    method: str
    display_name: str
    status: ModelStatus
    label: str | None = None
    confidence: float | None = None
    runtime_seconds: float | None = None
    error: str | None = None


class CompareResponse(BaseModel):
    text: str
    predictions: list[MethodPrediction]
    agreement: bool | None = Field(
        default=None, description="True if every AVAILABLE method's label agrees; null if fewer than 2 responded."
    )


class MethodInfo(BaseModel):
    method: str
    display_name: str
    status: ModelStatus
    description: str
