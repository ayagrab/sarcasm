"""DSPy + local Qwen adapter.

Loads the actual frozen, MIPROv2-optimized compiled program (instructions
+ few-shot demos, `results/<experiment_id>/compiled_program.json` --
written by `run_dspy_experiment`'s `program.save(...)` call, see
`src/classification/dspy_pipeline/run_dspy.py`) rather than reconstructing
an unoptimized `dspy.Predict` baseline -- DSPy's `Predict.load(path)`
populates a fresh `Predict(signature)`'s instructions/demos from the saved
JSON state, no retraining/recompilation involved. Falls back to the
unoptimized baseline only if the frozen program file is missing (reported
via `_init_error`, not hidden) -- e.g. if the registry names an
experiment_id whose `compiled_program.json` never got synced to this
machine.

Gated the same way as `qwen_adapter.py`: NOT_FROZEN_YET (model never
loaded) until Stage B Phase 2 marks `dspy` frozen in
`results/frozen_configs.json`.
"""
from __future__ import annotations

from app import PROJECT_ROOT
from app.adapters.base import BaseAdapter
from app.frozen_registry import frozen_experiment_id, is_frozen
from app.schemas import ModelStatus

CHECKPOINT = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_EXPERIMENT_ID = "EXP-008"


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

            program = dspy.Predict(build_signature())
            experiment_id = frozen_experiment_id(self.method) or DEFAULT_EXPERIMENT_ID
            compiled_path = PROJECT_ROOT / "results" / experiment_id / "compiled_program.json"
            if compiled_path.exists():
                program.load(str(compiled_path))
            self._program = program
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
