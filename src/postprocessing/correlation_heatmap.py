"""Spearman correlation between structural/NLP metrics and the quality score.

Reproduces the finding in docs/project_history.md (2026-07-16 meeting):
structural metrics (sentence length, word overlap) correlate weakly with
quality -- sarcasm interpretation quality is not explainable by simple
structural rules.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import settings
from src.common.file_utils import load_all_model_outputs, ensure_parent_dir

METRICS = ["classification", "length_difference", "overlap_ratio", "source_word_count", "target_word_count"]


def add_structural_metrics(df):
    """Add word-count and lexical-overlap columns derived from the raw text."""
    df = df.copy()
    df["source_word_count"] = df["sarcastic_sentence"].str.split().str.len().fillna(0)
    df["target_word_count"] = df["model_interpretation"].str.split().str.len().fillna(0)
    df["length_difference"] = df["target_word_count"] - df["source_word_count"]

    def overlap_ratio(row) -> float:
        source_words = {w.strip(".,!?\"'") for w in str(row["sarcastic_sentence"]).lower().split()}
        target_words = {w.strip(".,!?\"'") for w in str(row["model_interpretation"]).lower().split()}
        if not source_words:
            return 0.0
        return len(source_words & target_words) / len(source_words)

    df["overlap_ratio"] = df.apply(overlap_ratio, axis=1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Spearman correlation heatmap: structural metrics vs. quality score.")
    parser.add_argument("--outputs-dir", type=Path, default=settings.model_outputs_dir)
    parser.add_argument("--output", type=Path, default=settings.summaries_dir / "figures" / "nlp_metrics_correlation_heatmap.png")
    args = parser.parse_args()

    df = add_structural_metrics(load_all_model_outputs(args.outputs_dir))
    corr_matrix = df[METRICS].corr(method="spearman")
    print(corr_matrix.round(2))

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, linewidths=0.5, square=True)
    plt.title("Spearman Correlation Matrix: Metrics vs. Quality Score", fontsize=14, pad=20, fontweight="bold")
    plt.tight_layout()
    ensure_parent_dir(args.output)
    plt.savefig(args.output, dpi=300)
    print(f"Saved heatmap to {args.output}")


if __name__ == "__main__":
    main()
