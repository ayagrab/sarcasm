"""Phase 4, M1 leg: zero-transfer evaluation of the frozen TF-IDF+LR
approach (Part II's EXP-001 winning config, `configs/tfidf.json`) on
SIGN Test. No retraining on SIGN -- fits on Dataset A TRAIN only (exactly
as Part II did), predicts on SIGN Test (both roles). Local, CPU, seconds.

Run: `python -m src.sign.zero_transfer.run_m1_zero_transfer`
"""
from __future__ import annotations

import argparse

import pandas as pd

from config.classification_settings import classification_settings
from config.sign_settings import sign_settings
from src.classification.classical.tfidf_baseline import build_pipeline
from src.sign.data.load_sign import load_family_table
from src.sign.zero_transfer.io import save_zero_transfer_result

EXPERIMENT_ID = "EXP-SIGN-011"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 / M1: zero-transfer TF-IDF+LR on SIGN Test.")
    parser.add_argument("--seed", type=int, default=classification_settings.random_seed)
    args = parser.parse_args()

    train_df = pd.read_csv(classification_settings.splits_dir / "train.csv", encoding="utf-8-sig")
    sign_test = load_family_table("test")

    pipeline = build_pipeline(classifier="logreg", tfidf_variant="word_char_combo", seed=args.seed)
    pipeline.fit(train_df["text"], train_df["label"])
    preds = pipeline.predict(sign_test["text"])

    predictions = pd.DataFrame(
        {
            "example_id": sign_test["example_id"],
            "gold_label": sign_test["label"],
            "predicted_label": preds,
        }
    )

    config = {
        "experiment_id": EXPERIMENT_ID,
        "approach": "M1_tfidf_logreg_zero_transfer",
        "frozen_config_reused": "configs/tfidf.json (EXP-001)",
        "train_data": "Dataset A TRAIN only (data/splits/train.csv, n=6706) -- SIGN never seen during fit",
        "eval_data": "SIGN Test, all roles (data/sign/family_table_test.csv, n=1735)",
        "seed": args.seed,
    }
    save_zero_transfer_result(EXPERIMENT_ID, config, predictions, sign_test)


if __name__ == "__main__":
    main()
