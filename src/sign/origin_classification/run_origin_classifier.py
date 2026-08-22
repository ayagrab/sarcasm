"""Phase 3 (PROJECT_SUMMARY.md): dataset-of-origin diagnostic.

NOT a product classifier -- the only question is "how easily does a
simple model tell Dataset A text apart from SIGN text?" High accuracy is
evidence of domain shift, not a capability worth deploying.

Two conditions, run separately, to decompose the signal (Phase 2 found
SIGN text is almost always lowercase/unpunctuated while Dataset A is
not -- a trivial giveaway that has nothing to do with topic/content):

    EXP-SIGN-001 -- raw text, as committed in both corpora
    EXP-SIGN-002 -- case- and punctuation-normalized text (both corpora
                    lowercased, punctuation stripped, whitespace
                    collapsed) -- isolates topical/lexical-content signal
                    from surface formatting.

Train split: Dataset A train (6,706) + SIGN train, all roles (14,292).
Eval split: Dataset A test (1,340) + SIGN test, all roles (1,735) -- both
corpora's *own* held-out test partitions, so this never touches Dataset
A's frozen splits' purpose (still only read, never modified) and never
touches SIGN Dev/Test for anything but this one diagnostic evaluation.

TF-IDF (word 1-2gram + char 3-5gram, matching M1/EXP-001's winning
Part II config -- see `src.classification.classical.tfidf_baseline`,
imported not duplicated) + Logistic Regression, `class_weight="balanced"`
(SIGN train outnumbers Dataset A train ~2:1, would otherwise bias a
majority-class classifier).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from config.classification_settings import classification_settings
from config.sign_settings import sign_settings
from src.classification.classical.tfidf_baseline import TFIDF_PARAMS
from src.classification.evaluation.metrics import compute_metrics
from src.sign.data.load_sign import load_family_table

LABELS = ("dataset_a", "sign")
POSITIVE_LABEL = "sign"


def load_dataset_a_texts(split: str) -> pd.DataFrame:
    df = pd.read_csv(classification_settings.splits_dir / f"{split}.csv", encoding="utf-8-sig")
    out = df[["example_id", "text"]].copy()
    out["origin"] = "dataset_a"
    return out


def load_sign_texts(split: str) -> pd.DataFrame:
    df = load_family_table(split)
    out = df[["example_id", "text"]].copy()
    out["origin"] = "sign"
    return out


def build_origin_frame(split: str) -> pd.DataFrame:
    a = load_dataset_a_texts(split)
    s = load_sign_texts(split)
    combined = pd.concat([a, s], ignore_index=True)
    assert combined["example_id"].is_unique, "example_id collision between Dataset A and SIGN"
    return combined


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation (keep apostrophes so "n't"/"'s"
    survive, matching how SIGN's own text already handles contractions),
    collapse whitespace -- applied identically to both corpora."""
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_pipeline(seed: int) -> Pipeline:
    vectorizer = FeatureUnion(
        [
            ("word", TfidfVectorizer(**TFIDF_PARAMS["word_1_2"])),
            ("char", TfidfVectorizer(**TFIDF_PARAMS["char_3_5"])),
        ]
    )
    clf = LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced")
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def save_origin_experiment(experiment_id: str, config: dict, predictions: pd.DataFrame) -> Path:
    out_dir = sign_settings.results_dir / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions[["example_id", "gold_label", "predicted_label"]].to_csv(
        out_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    metrics = compute_metrics(predictions, labels=LABELS, positive_label=POSITIVE_LABEL)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    return out_dir


def run_condition(experiment_id: str, normalize: bool, seed: int) -> dict:
    train_df = build_origin_frame("train")
    test_df = build_origin_frame("test")

    if normalize:
        train_texts = train_df["text"].apply(normalize_text)
        test_texts = test_df["text"].apply(normalize_text)
    else:
        train_texts, test_texts = train_df["text"], test_df["text"]

    pipeline = build_pipeline(seed=seed)
    pipeline.fit(train_texts, train_df["origin"])
    preds = pipeline.predict(test_texts)

    predictions = pd.DataFrame(
        {
            "example_id": test_df["example_id"],
            "gold_label": test_df["origin"],
            "predicted_label": preds,
        }
    )
    config = {
        "experiment_id": experiment_id,
        "approach": "dataset_origin_diagnostic",
        "normalize_text": normalize,
        "seed": seed,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_dataset_a": int((train_df["origin"] == "dataset_a").sum()),
        "n_train_sign": int((train_df["origin"] == "sign").sum()),
        "n_test_dataset_a": int((test_df["origin"] == "dataset_a").sum()),
        "n_test_sign": int((test_df["origin"] == "sign").sum()),
        "vectorizer": "tfidf word(1,2) + char_wb(3,5), matches M1/EXP-001",
        "classifier": "LogisticRegression(class_weight='balanced')",
    }
    out_dir = save_origin_experiment(experiment_id, config, predictions)
    metrics = compute_metrics(predictions, labels=LABELS, positive_label=POSITIVE_LABEL)
    print(f"[{experiment_id}] normalize={normalize} accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"  -> {out_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: dataset-of-origin diagnostic classifier.")
    parser.add_argument("--seed", type=int, default=sign_settings.random_seed)
    args = parser.parse_args()

    run_condition("EXP-SIGN-001", normalize=False, seed=args.seed)
    run_condition("EXP-SIGN-002", normalize=True, seed=args.seed)


if __name__ == "__main__":
    main()
