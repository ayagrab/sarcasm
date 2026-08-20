"""Phase 10, M1 leg: interpretation-count ablation (SIGN_GENERALIZATION_PLAN.md,
Phase 10). RQ4. k=1 (Phase 8's condition B, EXP-SIGN-021) is reused, not
rerun. k=2/3/5 are new: Dataset A TRAIN (full, fixed) + Phase 7's k2/k3/k5
variants (100% of SIGN Train families, interpretations #1..k by rank),
imbalance handled via `class_weight="balanced"` (documented policy,
Phase 7's `.meta.json` sidecars) rather than row duplication -- the
simplest correct fix for a linear model. Same full 100% family set at
every k (not swept -- that's Phase 9's question, kept separate).

Local, CPU, seconds. Run:
`python -m src.sign.interp_ablation.run_m1_interp_ablation`
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from config.classification_settings import classification_settings
from config.sign_settings import sign_settings
from src.classification.classical.tfidf_baseline import TFIDF_PARAMS
from src.sign.domain_adaptation.io import save_domain_adaptation_result
from src.sign.data.load_sign import load_family_table

K_VALUES_TO_RUN = {2: "EXP-SIGN-033", 3: "EXP-SIGN-034", 5: "EXP-SIGN-035"}


def build_balanced_pipeline(seed: int) -> Pipeline:
    vectorizer = FeatureUnion(
        [
            ("word", TfidfVectorizer(**TFIDF_PARAMS["word_1_2"])),
            ("char", TfidfVectorizer(**TFIDF_PARAMS["char_3_5"])),
        ]
    )
    clf = LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced")
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def _predict(pipeline, texts: pd.Series, gold: pd.Series, example_ids: pd.Series) -> pd.DataFrame:
    preds = pipeline.predict(texts)
    return pd.DataFrame({"example_id": example_ids, "gold_label": gold, "predicted_label": preds})


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10 / M1: SIGN interpretation-count ablation (k=2/3/5).")
    parser.add_argument("--seed", type=int, default=classification_settings.random_seed)
    args = parser.parse_args()

    dataset_a_train = pd.read_csv(classification_settings.splits_dir / "train.csv", encoding="utf-8-sig")
    dataset_a_test = pd.read_csv(classification_settings.splits_dir / "test.csv", encoding="utf-8-sig")
    sign_test = load_family_table("test")

    for k, experiment_id in K_VALUES_TO_RUN.items():
        sign_k = pd.read_csv(sign_settings.processed_dir / "train_variants" / f"k{k}.csv", encoding="utf-8-sig")
        combined_train = pd.concat(
            [dataset_a_train[["text", "label"]], sign_k[["text", "label"]]], ignore_index=True
        )
        pipeline = build_balanced_pipeline(args.seed)
        pipeline.fit(combined_train["text"], combined_train["label"])

        save_domain_adaptation_result(
            experiment_id,
            config={
                "experiment_id": experiment_id,
                "approach": "M1_tfidf_logreg_interp_count_ablation",
                "k_interpretations": k,
                "imbalance_handling": "class_weight=balanced (LogisticRegression)",
                "train_data": f"Dataset A TRAIN (n={len(dataset_a_train)}) + SIGN Train k={k} "
                f"(n={len(sign_k)} rows, interpretations #1..{k} by rank)",
                "eval_data": "SIGN Test (all roles) + Dataset A TEST (forgetting check)",
                "seed": args.seed,
            },
            sign_test_predictions=_predict(pipeline, sign_test["text"], sign_test["label"], sign_test["example_id"]),
            dataset_a_test_predictions=_predict(
                pipeline, dataset_a_test["text"], dataset_a_test["label"], dataset_a_test["example_id"]
            ),
        )
        print(f"k={k}: {experiment_id} done ({len(sign_k)} SIGN rows)")


if __name__ == "__main__":
    main()
