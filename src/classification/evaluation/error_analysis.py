"""Cross-model disagreement / error analysis.

Loads `predictions.csv` from one or more experiment result directories
(same `eval_split` required for a fair comparison), merges them into one
wide per-example table, attaches canonical-dataset context (category, raw
text, label_conflict), and produces pairwise disagreement subsets (e.g.
"TF-IDF right, Qwen wrong"). Pure pandas, no GPU/model needed -- meant to
be re-run any time new experiment predictions become available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config.classification_settings import classification_settings


def load_predictions(experiment_id: str, model_label: str, results_dir: Path | None = None) -> pd.DataFrame:
    results_dir = results_dir or classification_settings.results_dir
    path = results_dir / experiment_id / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"No predictions.csv for experiment {experiment_id!r} at {path}")
    df = pd.read_csv(path)
    return df[["example_id", "gold_label", "predicted_label"]].rename(
        columns={"predicted_label": f"pred__{model_label}"}
    )


def build_disagreement_table(
    experiment_ids: dict[str, str], results_dir: Path | None = None
) -> pd.DataFrame:
    """`experiment_ids`: {model_label: experiment_id}. Returns one row per
    example_id (outer join -- an example missing from any experiment shows
    NaN for that model's column rather than being silently dropped)."""
    merged = None
    for label, exp_id in experiment_ids.items():
        df = load_predictions(exp_id, label, results_dir)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df.drop(columns=["gold_label"]), on="example_id", how="outer")
    return merged


def attach_dataset_context(disagreement_df: pd.DataFrame) -> pd.DataFrame:
    canonical = pd.read_csv(classification_settings.canonical_dataset_path)
    context_cols = ["example_id", "category", "text", "label_conflict"]
    return disagreement_df.merge(canonical[context_cols], on="example_id", how="left")


def pairwise_breakdown(df: pd.DataFrame, model_a: str, model_b: str) -> dict[str, pd.DataFrame]:
    """Four subsets: A-right/B-wrong, B-right/A-wrong, both wrong, both
    right -- the core "who disagrees and who's correct" comparison."""
    a_col, b_col = f"pred__{model_a}", f"pred__{model_b}"
    for col in (a_col, b_col):
        if col not in df.columns:
            raise KeyError(f"Model column {col!r} not found -- check the model_label used in build_disagreement_table")

    correct_a = df[a_col] == df["gold_label"]
    correct_b = df[b_col] == df["gold_label"]
    return {
        f"{model_a}_right_{model_b}_wrong": df[correct_a & ~correct_b],
        f"{model_b}_right_{model_a}_wrong": df[correct_b & ~correct_a],
        "both_wrong": df[~correct_a & ~correct_b],
        "both_right": df[correct_a & correct_b],
    }


def summarize_pairwise(df: pd.DataFrame, model_a: str, model_b: str) -> dict:
    breakdown = pairwise_breakdown(df, model_a, model_b)
    return {name: int(len(subset)) for name, subset in breakdown.items()}


def full_agreement_rate(df: pd.DataFrame, model_labels: list[str]) -> float:
    """Fraction of examples where every listed model predicts the same label."""
    cols = [f"pred__{m}" for m in model_labels]
    return float((df[cols].nunique(axis=1) == 1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="model_label=EXPERIMENT_ID pairs, e.g. tfidf=EXP-001-dev qwen_zero_shot=EXP-002",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    experiment_ids = dict(pair.split("=", 1) for pair in args.experiments)
    table = build_disagreement_table(experiment_ids)
    table = attach_dataset_context(table)

    labels = list(experiment_ids.keys())
    print(f"n_examples: {len(table)}")
    print(f"full agreement rate across {labels}: {full_agreement_rate(table, labels):.4f}")
    for i, model_a in enumerate(labels):
        for model_b in labels[i + 1 :]:
            print(f"\n{model_a} vs {model_b}:")
            print(json.dumps(summarize_pairwise(table, model_a, model_b), indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\nWrote full disagreement table to {args.output}")


if __name__ == "__main__":
    main()
