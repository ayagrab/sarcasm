"""Phase 4, M6 leg: zero-transfer evaluation of the frozen fine-tuned
DeBERTa-v3-base checkpoint (Part II's EXP-009 `best_checkpoint`, the
project's best Dataset-A model) on SIGN Test. No retraining, no
fine-tuning -- loads the exact checkpoint from disk and runs inference
only, mirroring `scripts/eval_frozen_checkpoint.py`'s eval-only pattern
(that script itself isn't reused directly since it's hardcoded to
Dataset A's splits directory; the model-loading logic it established is
reused via `src.classification.transformer.finetune`'s shared helpers).

CPU/MPS-feasible (~1,735 short examples) -- runs locally, no VM needed,
though nothing prevents running it on the VM in the same session as
M2-M5 for convenience.

Run: `python -m src.sign.zero_transfer.run_m6_zero_transfer`
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from config.classification_settings import classification_settings
from src.classification.transformer.finetune import ID2LABEL, build_dataset, select_device
from src.sign.data.load_sign import load_family_table
from src.sign.zero_transfer.io import save_zero_transfer_result

EXPERIMENT_ID = "EXP-SIGN-016"
DEFAULT_CHECKPOINT_DIR = classification_settings.models_dir / "EXP-009" / "best_checkpoint"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 / M6: zero-transfer frozen DeBERTa on SIGN Test.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device}")

    sign_test = load_family_table("test")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint_dir,
        num_labels=2,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
        use_safetensors=True,
        dtype=torch.float32,
    ).to(device)

    eval_dataset = build_dataset(tokenizer, sign_test, args.max_length)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/sign_eval_frozen_m6", per_device_eval_batch_size=32, fp16=False, report_to=[]),
    )
    raw_predictions = trainer.predict(eval_dataset)
    pred_ids = np.argmax(raw_predictions.predictions, axis=-1)

    predictions = pd.DataFrame(
        {
            "example_id": sign_test["example_id"].to_numpy(),
            "gold_label": sign_test["label"].to_numpy(),
            "predicted_label": [ID2LABEL[i] for i in pred_ids],
        }
    )

    config = {
        "experiment_id": EXPERIMENT_ID,
        "approach": "M6_finetuned_deberta_zero_transfer",
        "frozen_checkpoint_reused": str(args.checkpoint_dir) + " (EXP-009, Part II's best model)",
        "train_data": "NONE -- inference only, checkpoint already frozen from Dataset A TRAIN",
        "eval_data": "SIGN Test, all roles (data/sign/family_table_test.csv, n=1735)",
        "device": device,
        "max_length": args.max_length,
    }
    save_zero_transfer_result(EXPERIMENT_ID, config, predictions, sign_test)


if __name__ == "__main__":
    main()
