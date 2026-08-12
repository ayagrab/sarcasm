"""DSPy + local Qwen adapter.

Limitation, disclosed rather than hidden: this adapter serves the
`dspy.Predict` baseline program (same signature/adapter as
`run_dspy.py`'s `optimizer="predict"`), not a saved *optimized*
(BootstrapFewShot/MIPROv2) compiled program -- loading an arbitrary
frozen DSPy-optimized program back from `compiled_program.json` requires
matching DSPy's save/load API for the exact optimizer/version that
produced it, which is more machinery than is justified before Stage B has
even decided (Phase 2) which DSPy variant, if any, gets frozen. If
Phase 2 freezes an optimized variant instead of plain `Predict`, this
adapter should be extended to load that specific compiled program rather
than reconstructing the unoptimized baseline -- tracked here rather than
silently serving the wrong program.

Gated the same way as `qwen_adapter.py`: NOT_FROZEN_YET (model never
loaded) until Stage B Phase 2 marks `dspy` frozen in
`results/frozen_configs.json`.
"""
from __future__ import annotations

from app.adapters.base import BaseAdapter
from app.frozen_registry import is_frozen
from app.schemas import ModelStatus

CHECKPOINT = "Qwen/Qwen3-4B-Instruct-2507"


class DspyAdapter(BaseAdapter):
    method = "dspy"
    display_name = "Qwen + DSPy"
    description = (
        "The same local Qwen model, but prompted through a DSPy program instead of a "
        "hand-written prompt template -- DSPy structures the input/output contract and "
        "can optionally optimize it against labeled training examples."
    )

    def __init__(self) -> None:
        self._program = None
        self._init_error: str | None = None
        self._loaded = False

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import dspy

            from src.classification.dspy_pipeline.local_lm import LocalQwenLM
            from src.classification.dspy_pipeline.signatures import build_signature

            lm = LocalQwenLM(checkpoint=CHECKPOINT, temperature=0.0)
            dspy.configure(lm=lm)
            self._program = dspy.Predict(build_signature())
        except Exception as exc:  # no CUDA GPU, dspy not installed, etc.
            self._init_error = str(exc)

    def status(self) -> ModelStatus:
        if not is_frozen(self.method):
            return ModelStatus.NOT_FROZEN_YET
        self._lazy_load()
        if self._program is None:
            return ModelStatus.UNAVAILABLE
        return ModelStatus.AVAILABLE

    def _predict_raw(self, text: str) -> tuple[str, float | None]:
        prediction = self._program(sentence=text)
        return prediction.label, None
