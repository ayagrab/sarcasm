"""Phase 9, M1 leg: learning curve -- how much SIGN Train is needed
(PROJECT_SUMMARY.md, Phase 9). RQ3. Sweeps 0/10/25/50/75/100%
of SIGN Train families (primary balanced condition only -- interpretation
count is Phase 10's question, kept strictly separate), family-level
sampling via `sample_family_ids` (deterministic, seeded).

0% and 100% are NOT rerun here -- they're exactly Phase 4's zero-transfer
(EXP-SIGN-011) and Phase 8's condition B (EXP-SIGN-021), already
persisted. Only 10/25/50/75% are new runs. Same recipe as Phase 8's
condition B throughout: Dataset A TRAIN (full, fixed) + the SIGN Train
fraction, combined fit, evaluated on SIGN Test and Dataset A TEST.

Local, CPU, seconds per fraction. Run:
`python -m src.sign.learning_curve.run_m1_learning_curve`
"""
from __future__ import annotations

import argparse

import pandas as pd

from config.classification_settings import classification_settings
from config.sign_settings import sign_settings
from src.classification.classical.tfidf_baseline import build_pipeline
from src.sign.data.family_utils import sample_family_ids, select_families, select_primary_interpretation_per_family, unique_family_ids
from src.sign.data.load_sign import load_family_table
from src.sign.domain_adaptation.io import save_domain_adaptation_result

FRACTIONS_TO_RUN = {
    0.10: "EXP-SIGN-025",
    0.25: "EXP-SIGN-026",
    0.50: "EXP-SIGN-027",
    0.75: "EXP-SIGN-028",
}
REUSED_ENDPOINTS = {0.0: "EXP-SIGN-011", 1.0: "EXP-SIGN-021"}


def _predict(pipeline, texts: pd.Series, gold: pd.Series, example_ids: pd.Series) -> pd.DataFrame:
    preds = pipeline.predict(texts)
    return pd.DataFrame({"example_id": example_ids, "gold_label": gold, "predicted_label": preds})


def build_sign_fraction(train_table: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    family_ids = unique_family_ids(train_table)
    sampled_ids = sample_family_ids(family_ids, seed=seed, frac=frac)
    subset = select_families(train_table, sampled_ids)
    return select_primary_interpretation_per_family(subset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9 / M1: SIGN learning curve (10/25/50/75%).")
    parser.add_argument("--seed", type=int, default=classification_settings.random_seed)
    args = parser.parse_args()

    dataset_a_train = pd.read_csv(classification_settings.splits_dir / "train.csv", encoding="utf-8-sig")
    dataset_a_test = pd.read_csv(classification_settings.splits_dir / "test.csv", encoding="utf-8-sig")
    sign_test = load_family_table("test")
    train_table = load_family_table("train")

    for frac, experiment_id in FRACTIONS_TO_RUN.items():
        sign_fraction = build_sign_fraction(train_table, frac, args.seed)
        combined_train = pd.concat(
            [dataset_a_train[["text", "label"]], sign_fraction[["text", "label"]]], ignore_index=True
        )
        pipeline = build_pipeline(classifier="logreg", tfidf_variant="word_char_combo", seed=args.seed)
        pipeline.fit(combined_train["text"], combined_train["label"])

        save_domain_adaptation_result(
            experiment_id,
            config={
                "experiment_id": experiment_id,
                "approach": "M1_tfidf_logreg_learning_curve",
                "sign_train_fraction": frac,
                "n_sign_families_sampled": int(sign_fraction["family_id"].nunique()),
                "train_data": f"Dataset A TRAIN (n={len(dataset_a_train)}) + SIGN Train primary "
                f"fraction={frac} (n={len(sign_fraction)} rows, seed={args.seed})",
                "eval_data": "SIGN Test (all roles) + Dataset A TEST (forgetting check)",
                "seed": args.seed,
            },
            sign_test_predictions=_predict(pipeline, sign_test["text"], sign_test["label"], sign_test["example_id"]),
            dataset_a_test_predictions=_predict(
                pipeline, dataset_a_test["text"], dataset_a_test["label"], dataset_a_test["example_id"]
            ),
        )
        print(f"frac={frac}: {experiment_id} done ({len(sign_fraction)} SIGN rows)")


if __name__ == "__main__":
    main()
