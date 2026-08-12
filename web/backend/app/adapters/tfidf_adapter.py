"""TF-IDF + Logistic Regression adapter. Reuses `build_pipeline` /
`load_split` from the existing classical baseline module directly --
fits the exact frozen configuration (`configs/tfidf.json`: `logreg` /
`word_char_combo`, seed 42) once on TRAIN at startup, then serves
in-memory predictions. Fitting takes a few seconds on CPU; not worth
persisting a pickle for a model this cheap to refit deterministically.
"""
from __future__ import annotations

import json

from app import PROJECT_ROOT
from app.adapters.base import BaseAdapter
from app.frozen_registry import is_frozen
from app.schemas import ModelStatus

METHOD = "tfidf"
CONFIG_PATH = PROJECT_ROOT / "configs" / "tfidf.json"


class TfidfAdapter(BaseAdapter):
    method = METHOD
    display_name = "TF-IDF + Logistic Regression"
    description = (
        "A classical baseline: text is converted into word- and character-"
        "n-gram TF-IDF features, then classified with logistic regression. "
        "No neural network, no context understanding beyond word/character "
        "statistics -- fast and surprisingly competitive on this corpus."
    )

    def __init__(self) -> None:
        self._pipeline = None
        self._init_error: str | None = None
        self._try_fit()

    def _try_fit(self) -> None:
        try:
            from src.classification.classical.tfidf_baseline import build_pipeline, load_split

            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            train_df = load_split("train")
            pipeline = build_pipeline(
                classifier=config["classifier"],
                tfidf_variant=config["tfidf_variant"],
                seed=config["seed"],
            )
            pipeline.fit(train_df["text"], train_df["label"])
            self._pipeline = pipeline
        except Exception as exc:  # startup must never crash the app over one adapter
            self._init_error = str(exc)

    def status(self) -> ModelStatus:
        if not is_frozen(self.method):
            return ModelStatus.NOT_FROZEN_YET
        if self._pipeline is None:
            return ModelStatus.UNAVAILABLE
        return ModelStatus.AVAILABLE

    def _predict_raw(self, text: str) -> tuple[str, float | None]:
        label = self._pipeline.predict([text])[0]
        confidence = None
        if hasattr(self._pipeline.named_steps["clf"], "predict_proba"):
            proba = self._pipeline.predict_proba([text])[0]
            confidence = float(proba.max())
        return label, confidence
