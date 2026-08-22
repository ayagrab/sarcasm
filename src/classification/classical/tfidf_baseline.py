"""Classical baseline: TF-IDF -> Logistic Regression (and Linear SVM).

Trains on the canonical TRAIN split. Configuration selection (which
vectorizer variant, which classifier) happens by evaluating on DEV only;
the chosen frozen configuration is evaluated on TEST exactly once. See
PROJECT_SUMMARY.md, EXP-001 series, for the sweep and the final result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from config.classification_settings import classification_settings
from src.classification.evaluation.io import save_experiment_artifacts
from src.classification.evaluation.metrics import compute_metrics

TFIDF_PARAMS = {
    "word_1_2": {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "max_features": 50000},
    "char_3_5": {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "max_features": 50000},
}
TFIDF_VARIANTS = ("word_1_2", "char_3_5", "word_char_combo")
CLASSIFIERS = ("logreg", "linear_svm")


def load_split(split_name: str, splits_dir: Path | None = None) -> pd.DataFrame:
    splits_dir = splits_dir or classification_settings.splits_dir
    path = splits_dir / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run make_splits.py first")
    return pd.read_csv(path)


def _build_vectorizer(tfidf_variant: str):
    if tfidf_variant == "word_char_combo":
        return FeatureUnion(
            [
                ("word", TfidfVectorizer(**TFIDF_PARAMS["word_1_2"])),
                ("char", TfidfVectorizer(**TFIDF_PARAMS["char_3_5"])),
            ]
        )
    return TfidfVectorizer(**TFIDF_PARAMS[tfidf_variant])


def build_pipeline(classifier: str, tfidf_variant: str, seed: int) -> Pipeline:
    if classifier == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    elif classifier == "linear_svm":
        clf = LinearSVC(random_state=seed)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    return Pipeline([("tfidf", _build_vectorizer(tfidf_variant)), ("clf", clf)])


def run_tfidf_experiment(
    experiment_id: str,
    classifier: str = "logreg",
    tfidf_variant: str = "word_1_2",
    eval_split: str = "dev",
    seed: int | None = None,
    save_artifacts: bool = True,
) -> dict:
    seed = classification_settings.random_seed if seed is None else seed
    train_df = load_split("train")
    eval_df = load_split(eval_split)

    pipeline = build_pipeline(classifier, tfidf_variant, seed)
    pipeline.fit(train_df["text"], train_df["label"])

    pred_labels = pipeline.predict(eval_df["text"])
    predictions = pd.DataFrame(
        {
            "example_id": eval_df["example_id"],
            "gold_label": eval_df["label"],
            "predicted_label": pred_labels,
        }
    )

    clf_step = pipeline.named_steps["clf"]
    if hasattr(clf_step, "predict_proba"):
        proba = pipeline.predict_proba(eval_df["text"])
        predictions["confidence"] = proba.max(axis=1)

    config = {
        "experiment_id": experiment_id,
        "approach": "M1_tfidf_classical",
        "classifier": classifier,
        "tfidf_variant": tfidf_variant,
        "eval_split": eval_split,
        "seed": seed,
        "n_train": int(len(train_df)),
        "n_eval": int(len(eval_df)),
    }

    metrics = compute_metrics(predictions)
    out_dir = None
    if save_artifacts:
        out_dir = str(save_experiment_artifacts(experiment_id, config, predictions))

    return {"config": config, "metrics": metrics, "out_dir": out_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--classifier", choices=CLASSIFIERS, default="logreg")
    parser.add_argument("--tfidf-variant", choices=TFIDF_VARIANTS, default="word_1_2")
    parser.add_argument("--eval-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result = run_tfidf_experiment(
        args.experiment_id, args.classifier, args.tfidf_variant, args.eval_split, args.seed
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"\nArtifacts written to {result['out_dir']}")


if __name__ == "__main__":
    main()
