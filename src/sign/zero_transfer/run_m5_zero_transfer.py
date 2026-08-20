"""Phase 4, M5 leg: zero-transfer evaluation of the frozen, already
-compiled MIPROv2 DSPy program (Part II's EXP-008 winning config) on SIGN
Test.

Critically, this is **inference only, not recompilation**:
`results/EXP-008/compiled_program.json` (the frozen instructions + 4
bootstrapped demos MIPROv2 settled on, optimized against Dataset A
TRAIN/DEV) is loaded from disk via `dspy.Predict.load(...)` and reused
as-is. Part II's own M5-TEST run had no such load path and paid the full
~2h48m compile+eval cost every time (see EXPERIMENT_LOG.md / project
memory) -- that limitation doesn't apply here since the *frozen* program
state was already saved to disk when EXP-008 first ran, and DSPy's
programs are portable (the compiled state has no reference to Dataset A
except the human-readable demo text baked into it at compile time).

Requires the Azure VM (CUDA), same `LocalQwenLM` backend as M2-M4.

Run: `python -m src.sign.zero_transfer.run_m5_zero_transfer`
"""
from __future__ import annotations

import argparse
from pathlib import Path

import dspy
import pandas as pd
from tqdm import tqdm

from config.classification_settings import classification_settings
from src.classification.dspy_pipeline.local_lm import LocalQwenLM
from src.classification.dspy_pipeline.signatures import build_signature
from src.sign.data.load_sign import load_family_table
from src.sign.zero_transfer.io import save_zero_transfer_result

EXPERIMENT_ID = "EXP-SIGN-015"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_COMPILED_PROGRAM = classification_settings.project_root / "results" / "EXP-008" / "compiled_program.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 / M5: zero-transfer frozen MIPROv2 DSPy program on SIGN Test.")
    parser.add_argument("--compiled-program", type=Path, default=DEFAULT_COMPILED_PROGRAM)
    args = parser.parse_args()

    if not args.compiled_program.exists():
        raise FileNotFoundError(
            f"{args.compiled_program} not found -- copy results/EXP-008/compiled_program.json "
            "from the local Mac to the VM before running this."
        )

    lm = LocalQwenLM(checkpoint=MODEL, temperature=0.0)
    dspy.configure(lm=lm)

    signature = build_signature()
    program = dspy.Predict(signature)
    program.load(str(args.compiled_program))
    print(f"Loaded frozen compiled program from {args.compiled_program} (no recompilation).")

    sign_test = load_family_table("test")

    results = []
    for row in tqdm(sign_test.to_dict("records"), desc="M5 (frozen MIPROv2, inference only)"):
        prediction = program(sentence=row["text"])
        results.append({"example_id": row["example_id"], "gold_label": row["label"], "predicted_label": prediction.label})

    predictions = pd.DataFrame(results)
    config = {
        "experiment_id": EXPERIMENT_ID,
        "approach": "M5_dspy_mipro_v2_zero_transfer_inference_only",
        "frozen_program_reused": str(args.compiled_program) + " (EXP-008, NOT recompiled)",
        "model": MODEL,
        "provider": "local_hf",
        "temperature": 0.0,
        "train_data": "NONE -- inference only against the already-compiled frozen program",
        "eval_data": "SIGN Test, all roles (data/sign/family_table_test.csv, n=1735)",
        "dspy_version": getattr(dspy, "__version__", "unknown"),
    }
    save_zero_transfer_result(EXPERIMENT_ID, config, predictions, sign_test)


if __name__ == "__main__":
    main()
