"""Evaluate an already-trained (frozen) M6 DeBERTa checkpoint on a given
split, without retraining.

`finetune.py`'s `finetune_and_evaluate` always calls `trainer.train()` --
there is no "load checkpoint, just eval" path. Reusing it for Phase 2's
sealed-TEST evaluation would silently train a *second* model from scratch
(same config/seed, but not guaranteed bit-identical to the actual frozen
EXP-009 checkpoint that was reviewed and recorded) rather than evaluating
the frozen artifact itself -- this script exists specifically to avoid
that gap, evaluating the exact checkpoint on disk.

Usage:
    python -m scripts.eval_frozen_checkpoint \
        --checkpoint-dir models/EXP-009/best_checkpoint \
        --split test --experiment-id EXP-009-TEST --max-length 128
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from config.classification_settings import classification_settings
from src.classification.evaluation.io import save_experiment_artifacts
from src.classification.evaluation.metrics import compute_metrics
from src.classification.transformer.finetune import ID2LABEL, LABEL2ID, build_dataset, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--source-experiment-id", default="EXP-009", help="Experiment ID the checkpoint was frozen from, for the record")
    args = parser.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    device = select_device()
    eval_df = pd.read_csv(classification_settings.splits_dir / f"{args.split}.csv")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint_dir,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        use_safetensors=True,
        dtype=torch.float32,
    ).to(device)

    eval_dataset = build_dataset(tokenizer, eval_df, args.max_length)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/eval_frozen_checkpoint", per_device_eval_batch_size=32, fp16=False, report_to=[]),
    )
    raw_predictions = trainer.predict(eval_dataset)
    pred_ids = np.argmax(raw_predictions.predictions, axis=-1)
    confidences = torch.softmax(torch.tensor(raw_predictions.predictions), dim=-1).max(dim=-1).values.numpy()

    predictions = pd.DataFrame(
        {
            "example_id": eval_df["example_id"].to_numpy(),
            "gold_label": eval_df["label"].to_numpy(),
            "predicted_label": [ID2LABEL[i] for i in pred_ids],
            "confidence": confidences,
        }
    )

    config = {
        "approach": "M6_finetuned_transformer_eval_only",
        "checkpoint_dir": args.checkpoint_dir,
        "source_experiment_id": args.source_experiment_id,
        "eval_split": args.split,
        "max_length": args.max_length,
        "device": device,
        "n_eval": int(len(eval_df)),
    }
    out_dir = save_experiment_artifacts(args.experiment_id, config, predictions)
    metrics = compute_metrics(predictions)
    print(metrics)
    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main()
