"""Phase 8, M6 leg: domain adaptation for DeBERTa-v3-base
(PROJECT_SUMMARY.md, Phase 8). Conditions B and C only --
condition A is Phase 4's already-frozen zero-transfer result
(`EXP-SIGN-016`, from the fully-frozen `EXP-009` checkpoint), reused not
rerun.

- (B) EXP-SIGN-023: Dataset A TRAIN + SIGN Train primary condition
  (original + interpretation #1, Phase 7), combined, fine-tuned once.
  Early-stopped on Dataset A's own Dev split, matching EXP-009's exact
  methodology (same hyperparameters throughout, for comparability).
- (C) EXP-SIGN-024: SIGN Train primary condition only, fine-tuned alone
  -- Dataset A never seen. Early-stopped on SIGN's own Dev split
  (primary condition), since using Dataset A Dev here would partially
  undermine condition C's "Dataset A never seen" framing.

Both evaluated on SIGN Test (RQ2/RQ3) and Dataset A's own held-out TEST
(catastrophic-forgetting check). Reuses `finetune.py`'s low-level
building blocks (tokenization, dataset wrapping, device selection,
TransformerConfig) -- not `finetune_and_evaluate` itself, since that
function hardcodes reading Dataset-A-only train/dev files, which doesn't
fit SIGN's combined/custom train+dev sets here.

VM required (GPU fine-tuning). Run:
`python -m src.sign.domain_adaptation.run_m6_domain_adaptation`
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from config.classification_settings import classification_settings
from config.sign_settings import sign_settings
from src.classification.transformer.finetune import (
    ID2LABEL,
    LABEL2ID,
    SarcasmTorchDataset,
    TransformerConfig,
    build_dataset,
    compute_trainer_metrics,
    select_device,
    tokenize_texts,
)
from src.sign.data.family_utils import select_primary_interpretation_per_family
from src.sign.data.load_sign import load_family_table
from src.sign.domain_adaptation.io import save_domain_adaptation_result

EXPERIMENT_ID_B = "EXP-SIGN-023"
EXPERIMENT_ID_C = "EXP-SIGN-024"

# Matches EXP-009 exactly (results/EXP-009/config.json) for comparability
# -- only the train/dev data differ between conditions, not the recipe.
BASE_CONFIG_KWARGS = dict(
    checkpoint="microsoft/deberta-v3-base",
    max_length=128,
    learning_rate=1e-5,
    train_batch_size=16,
    eval_batch_size=32,
    num_epochs=5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    seed=classification_settings.random_seed,
    use_fast_tokenizer=False,
    use_safetensors=True,
    early_stopping_patience=2,
    fp16=True,
)


def _predict(trainer, tokenizer, df: pd.DataFrame, max_length: int) -> pd.DataFrame:
    encodings = tokenize_texts(tokenizer, df["text"].tolist(), max_length)
    dataset = SarcasmTorchDataset(encodings, [LABEL2ID[l] for l in df["label"]])
    raw = trainer.predict(dataset)
    pred_ids = np.argmax(raw.predictions, axis=-1)
    return pd.DataFrame(
        {
            "example_id": df["example_id"].to_numpy(),
            "gold_label": df["label"].to_numpy(),
            "predicted_label": [ID2LABEL[i] for i in pred_ids],
        }
    )


def run_condition(experiment_id: str, condition_label: str, train_df: pd.DataFrame, dev_df: pd.DataFrame) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

    config = TransformerConfig(experiment_id=experiment_id, **BASE_CONFIG_KWARGS)
    set_seed(config.seed)
    device = select_device()

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

    output_dir = classification_settings.models_dir / experiment_id
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

    sign_test = load_family_table("test")
    dataset_a_test = pd.read_csv(classification_settings.splits_dir / "test.csv", encoding="utf-8-sig")

    sign_test_predictions = _predict(trainer, tokenizer, sign_test, config.max_length)
    dataset_a_test_predictions = _predict(trainer, tokenizer, dataset_a_test, config.max_length)

    experiment_config = asdict(config)
    experiment_config.update(
        {
            "approach": f"M6_deberta_domain_adaptation_condition_{condition_label}",
            "condition": condition_label,
            "device": device,
            "n_train": int(len(train_df)),
            "n_dev": int(len(dev_df)),
            "best_checkpoint_dir": str(best_checkpoint_dir),
        }
    )
    save_domain_adaptation_result(experiment_id, experiment_config, sign_test_predictions, dataset_a_test_predictions)


def main() -> None:
    dataset_a_train = pd.read_csv(classification_settings.splits_dir / "train.csv", encoding="utf-8-sig")
    dataset_a_dev = pd.read_csv(classification_settings.splits_dir / "dev.csv", encoding="utf-8-sig")
    sign_train_primary = pd.read_csv(
        sign_settings.processed_dir / "train_variants" / "primary.csv", encoding="utf-8-sig"
    )
    sign_dev_primary = select_primary_interpretation_per_family(load_family_table("dev"))

    # Condition B: Dataset A TRAIN + SIGN Train primary, combined; early-stop on Dataset A Dev
    combined_train = pd.concat(
        [dataset_a_train[["example_id", "text", "label"]], sign_train_primary[["example_id", "text", "label"]]],
        ignore_index=True,
    )
    print(f"Condition B: {len(combined_train)} train, {len(dataset_a_dev)} dev (Dataset A)")
    run_condition(EXPERIMENT_ID_B, "B: Dataset A TRAIN + SIGN Train primary", combined_train, dataset_a_dev)

    # Condition C: SIGN Train primary only; early-stop on SIGN Dev primary
    print(f"Condition C: {len(sign_train_primary)} train, {len(sign_dev_primary)} dev (SIGN)")
    run_condition(
        EXPERIMENT_ID_C,
        "C: SIGN Train primary only",
        sign_train_primary[["example_id", "text", "label"]],
        sign_dev_primary[["example_id", "text", "label"]],
    )


if __name__ == "__main__":
    main()
