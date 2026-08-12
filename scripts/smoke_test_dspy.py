"""M5 gate: smoke-test the LocalQwenLM DSPy adapter on a handful of real
TRAIN/DEV examples with an unoptimized dspy.Predict program, before
spending a full-DEV (1,340-call) EXP-006 run or any optimizer compile on
it. Exercises the same adapter/signature/local Qwen client path EXP-006/
007/008 use, just on ~5 examples instead of hundreds.

Usage (on the VM, inside the activated venv, HF_HOME set):
    python scripts/smoke_test_dspy.py
Exits non-zero on any adapter/parsing error.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.classification_settings import classification_settings
from src.classification.dspy_pipeline.local_lm import LocalQwenLM
from src.classification.dspy_pipeline.signatures import build_signature

N_EXAMPLES = 5


def main() -> None:
    import dspy

    lm = LocalQwenLM(checkpoint="Qwen/Qwen3-4B-Instruct-2507", temperature=0.0)
    dspy.configure(lm=lm)

    signature = build_signature()
    program = dspy.Predict(signature)

    dev_df = pd.read_csv(classification_settings.splits_dir / "dev.csv").head(N_EXAMPLES)

    n_ok = 0
    for _, row in dev_df.iterrows():
        prediction = program(sentence=row["text"])
        label = getattr(prediction, "label", None)
        ok = label in ("sarcastic", "not_sarcastic")
        print(f"[{'OK' if ok else 'FAIL'}] example_id={row['example_id']} gold={row['label']} pred={label!r}")
        if not ok:
            raise RuntimeError(f"Adapter returned an invalid/unparseable label: {label!r}")
        n_ok += 1

    print(f"\nDSPy smoke test PASSED: {n_ok}/{N_EXAMPLES} examples produced a valid label via LocalQwenLM.")


if __name__ == "__main__":
    main()
