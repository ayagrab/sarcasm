"""Phase 8, M1 leg: domain adaptation for TF-IDF+LR (SIGN_GENERALIZATION_PLAN.md,
Phase 8). Conditions B and C only -- condition A is Phase 4's already-
frozen zero-transfer result (`EXP-SIGN-011`) and Part II's own frozen
Dataset-A-TEST result (`results/EXP-001/metrics.json`), reused not rerun.

- (B) EXP-SIGN-021: Dataset A TRAIN + SIGN Train primary condition
  (original + interpretation #1, Phase 7), combined, fit once.
- (C) EXP-SIGN-022: SIGN Train primary condition only, fit alone.

Both evaluated on SIGN Test (RQ2/RQ3: does SIGN exposure help) and on
Dataset A's own held-out TEST (catastrophic-forgetting check). Local,
CPU, seconds -- no VM needed for M1.

Run: `python -m src.sign.domain_adaptation.run_m1_domain_adaptation`
"""
from __future__ import annotations

import argparse

import pandas as pd

from config.classification_settings import classification_settings
from config.sign_settings import sign_settings
from src.classification.classical.tfidf_baseline import build_pipeline
from src.sign.data.load_sign import load_family_table
from src.sign.domain_adaptation.io import save_domain_adaptation_result

EXPERIMENT_ID_B = "EXP-SIGN-021"
EXPERIMENT_ID_C = "EXP-SIGN-022"


def _predict(pipeline, texts: pd.Series, gold: pd.Series, example_ids: pd.Series) -> pd.DataFrame:
    preds = pipeline.predict(texts)
    return pd.DataFrame({"example_id": example_ids, "gold_label": gold, "predicted_label": preds})


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 / M1: SIGN domain adaptation, conditions B and C.")
    parser.add_argument("--seed", type=int, default=classification_settings.random_seed)
    args = parser.parse_args()

    dataset_a_train = pd.read_csv(classification_settings.splits_dir / "train.csv", encoding="utf-8-sig")
    dataset_a_test = pd.read_csv(classification_settings.splits_dir / "test.csv", encoding="utf-8-sig")
    sign_test = load_family_table("test")
    sign_train_primary = pd.read_csv(sign_settings.processed_dir / "train_variants" / "primary.csv", encoding="utf-8-sig")

    # Condition B: Dataset A TRAIN + SIGN Train primary, combined
    combined_train = pd.concat(
        [dataset_a_train[["text", "label"]], sign_train_primary[["text", "label"]]], ignore_index=True
    )
    pipeline_b = build_pipeline(classifier="logreg", tfidf_variant="word_char_combo", seed=args.seed)
    pipeline_b.fit(combined_train["text"], combined_train["label"])
    save_domain_adaptation_result(
        EXPERIMENT_ID_B,
        config={
            "experiment_id": EXPERIMENT_ID_B,
            "approach": "M1_tfidf_logreg_domain_adaptation_condition_B",
            "condition": "B: Dataset A TRAIN + SIGN Train primary (combined fit)",
            "train_data": f"Dataset A TRAIN (n={len(dataset_a_train)}) + SIGN Train primary "
            f"(n={len(sign_train_primary)}, original+interp#1, Phase 7)",
            "eval_data": "SIGN Test (all roles) + Dataset A TEST (forgetting check)",
            "seed": args.seed,
        },
        sign_test_predictions=_predict(pipeline_b, sign_test["text"], sign_test["label"], sign_test["example_id"]),
        dataset_a_test_predictions=_predict(
            pipeline_b, dataset_a_test["text"], dataset_a_test["label"], dataset_a_test["example_id"]
        ),
    )

    # Condition C: SIGN Train primary only
    pipeline_c = build_pipeline(classifier="logreg", tfidf_variant="word_char_combo", seed=args.seed)
    pipeline_c.fit(sign_train_primary["text"], sign_train_primary["label"])
    save_domain_adaptation_result(
        EXPERIMENT_ID_C,
        config={
            "experiment_id": EXPERIMENT_ID_C,
            "approach": "M1_tfidf_logreg_domain_adaptation_condition_C",
            "condition": "C: SIGN Train primary only",
            "train_data": f"SIGN Train primary only (n={len(sign_train_primary)}, original+interp#1, Phase 7) "
            "-- Dataset A never seen during fit",
            "eval_data": "SIGN Test (all roles) + Dataset A TEST (forgetting check)",
            "seed": args.seed,
        },
        sign_test_predictions=_predict(pipeline_c, sign_test["text"], sign_test["label"], sign_test["example_id"]),
        dataset_a_test_predictions=_predict(
            pipeline_c, dataset_a_test["text"], dataset_a_test["label"], dataset_a_test["example_id"]
        ),
    )


if __name__ == "__main__":
    main()
