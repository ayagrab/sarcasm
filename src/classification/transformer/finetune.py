"""Fine-tune a pretrained English encoder for sarcastic vs. not-sarcastic classification.

Config-driven: checkpoint, tokenizer, max sequence length, batch size,
learning rate, epochs, warmup, weight decay, seed, device, mixed
precision. Uses `transformers.Trainer` with early stopping on dev Macro F1
(the same primary metric used everywhere else in this project).

Used for M6's full DEV training run (EXP-009, `microsoft/deberta-v3-base`)
and its sealed-TEST evaluation (see `PROJECT_SUMMARY.md`). Requires
`accelerate` and `sentencepiece` -- see `requirements-classification.txt`.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.classification_settings import classification_settings
from src.classification.evaluation.io import save_experiment_artifacts
from src.classification.evaluation.metrics import compute_metrics

LABEL2ID = {"not_sarcastic": 0, "sarcastic": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

APPROACH_ID = "M6_finetuned_transformer"

# Candidate English encoders considered for this task (see PROJECT_SUMMARY.md
# "Stage A readiness report" for why `roberta-base` is the recommended
# first checkpoint). Kept here, not hardcoded elsewhere, so the choice is
# visible and easy to extend.
CANDIDATE_CHECKPOINTS = (
    "roberta-base",
    "microsoft/deberta-v3-base",
    "bert-base-uncased",
)


@dataclass
class TransformerConfig:
    experiment_id: str
    checkpoint: str = "roberta-base"
    max_length: int = 128
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = classification_settings.random_seed
    # DeBERTa-v3's tokenizer must be loaded as slow/SentencePiece
    # (use_fast_tokenizer=False); RoBERTa/BERT-family checkpoints use the
    # fast tokenizer (default True). See PROJECT_SUMMARY.md, Stage B, M6.
    use_fast_tokenizer: bool = True
    # transformers on torch>=2.5 blocks unsafe torch.load paths for
    # certain older-format checkpoints -- load via safetensors explicitly.
    use_safetensors: bool = True
    # Smoke-test support (task: "first run a tiny overfit/smoke test,
    # e.g. 32-100 training examples" before the full run). None = use the
    # full split.
    max_train_examples: int | None = None
    max_dev_examples: int | None = None
    early_stopping_patience: int = 2
    fp16: bool = False  # only meaningful on CUDA; left False by default for MPS/CPU


class SarcasmTorchDataset(Dataset):
    """Wraps pre-tokenized encodings + integer labels for `Trainer`."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tokenize_texts(tokenizer, texts: list[str], max_length: int) -> dict:
    return tokenizer(list(texts), truncation=True, padding="max_length", max_length=max_length)


def build_dataset(tokenizer, df: pd.DataFrame, max_length: int) -> SarcasmTorchDataset:
    encodings = tokenize_texts(tokenizer, df["text"].tolist(), max_length)
    labels = [LABEL2ID[label] for label in df["label"]]
    return SarcasmTorchDataset(encodings, labels)


def compute_trainer_metrics(eval_pred) -> dict:
    """Metric callback passed to `Trainer` for early stopping / checkpoint
    selection during training -- deliberately Macro F1, matching the
    project-wide primary selection metric in `evaluation/metrics.py`."""
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "accuracy": float((preds == labels).mean()),
    }


def finetune_and_evaluate(config: TransformerConfig, eval_split: str = "dev") -> dict:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(config.seed)
    device = select_device()

    splits_dir = classification_settings.splits_dir
    train_df = pd.read_csv(splits_dir / "train.csv")
    dev_df = pd.read_csv(splits_dir / "dev.csv")
    eval_df = pd.read_csv(splits_dir / f"{eval_split}.csv")

    if config.max_train_examples is not None:
        train_df = train_df.sample(n=min(config.max_train_examples, len(train_df)), random_state=config.seed)
    if config.max_dev_examples is not None:
        dev_df = dev_df.sample(n=min(config.max_dev_examples, len(dev_df)), random_state=config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, use_fast=config.use_fast_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        use_safetensors=config.use_safetensors,
        dtype=torch.float32,
    ).to(device)

    train_dataset = build_dataset(tokenizer, train_df, config.max_length)
    dev_dataset = build_dataset(tokenizer, dev_df, config.max_length)

    output_dir = classification_settings.models_dir / config.experiment_id
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_ratio,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=config.fp16 and device == "cuda",
        seed=config.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    trainer.train()

    best_checkpoint_dir = output_dir / "best_checkpoint"
    trainer.save_model(str(best_checkpoint_dir))
    tokenizer.save_pretrained(str(best_checkpoint_dir))

    eval_encodings = tokenize_texts(tokenizer, eval_df["text"].tolist(), config.max_length)
    eval_dataset = SarcasmTorchDataset(eval_encodings, [LABEL2ID[l] for l in eval_df["label"]])
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

    experiment_config = asdict(config)
    experiment_config.update(
        {
            "approach": APPROACH_ID,
            "eval_split": eval_split,
            "device": device,
            "n_train": int(len(train_df)),
            "n_dev": int(len(dev_df)),
            "n_eval": int(len(eval_df)),
            "best_checkpoint_dir": str(best_checkpoint_dir),
        }
    )

    out_dir = save_experiment_artifacts(config.experiment_id, experiment_config, predictions)
    metrics = compute_metrics(predictions)
    return {"config": experiment_config, "metrics": metrics, "out_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--checkpoint", default="roberta-base", choices=CANDIDATE_CHECKPOINTS)
    parser.add_argument("--eval-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=classification_settings.random_seed)
    parser.add_argument("--use-fast-tokenizer", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--use-safetensors", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--max-train-examples", type=int, default=None, help="Smoke-test: subsample TRAIN")
    parser.add_argument("--max-dev-examples", type=int, default=None, help="Smoke-test: subsample DEV")
    args = parser.parse_args()

    config = TransformerConfig(
        experiment_id=args.experiment_id,
        checkpoint=args.checkpoint,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        seed=args.seed,
        use_fast_tokenizer=args.use_fast_tokenizer,
        use_safetensors=args.use_safetensors,
        max_train_examples=args.max_train_examples,
        max_dev_examples=args.max_dev_examples,
    )
    result = finetune_and_evaluate(config, eval_split=args.eval_split)
    print(json.dumps(result["metrics"], indent=2))
    print(f"\nArtifacts written to {result['out_dir']}")


if __name__ == "__main__":
    main()
