"""Structural/linguistic plots: sentence length and word-overlap patterns.

Reproduces the three rewriting-strategy findings in
docs/meeting_notes_summary.md (2026-07-16 meeting): Gemini expands sentences,
Nvidia keeps length stable, Liquid drastically shortens and loses word
overlap with the source; and high-quality (score 3) translations tend to
keep moderate-to-high word overlap with the source.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import settings
from src.common.file_utils import ensure_parent_dir
from src.postprocessing.correlation_heatmap import add_structural_metrics
from src.common.file_utils import load_all_model_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot structural/linguistic rewriting patterns per model and prompt.")
    parser.add_argument("--outputs-dir", type=Path, default=settings.model_outputs_dir)
    parser.add_argument("--output-dir", type=Path, default=settings.summaries_dir / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("ggplot")

    df = add_structural_metrics(load_all_model_outputs(args.outputs_dir))

    # Plot 1: source vs. translated sentence length, per model
    plt.figure(figsize=(11, 6))
    length_melted = df.melt(
        id_vars=["model"], value_vars=["source_word_count", "target_word_count"],
        var_name="sentence_type", value_name="word_count",
    )
    length_melted["sentence_type"] = length_melted["sentence_type"].map(
        {"source_word_count": "Sarcastic (Source)", "target_word_count": "Neutral (Interpretation)"}
    )
    sns.barplot(data=length_melted, x="model", y="word_count", hue="sentence_type", palette="muted")
    plt.title("Structural Analysis: Average Word Count (Source vs. Translation)", fontsize=14)
    plt.ylabel("Average Number of Words")
    plt.xlabel("Model")
    out1 = args.output_dir / "linguistic_1_length_comparison.png"
    ensure_parent_dir(out1)
    plt.savefig(out1, bbox_inches="tight")
    plt.close()

    # Plot 2: word overlap ratio across prompts and models
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=df, x="prompt", y="overlap_ratio", hue="model", palette="Set2")
    plt.title("Lexical Overlap Ratio across Prompts and Models", fontsize=14)
    plt.ylabel("Overlap Ratio (Shared Words / Source Words)")
    plt.xlabel("Prompt Number")
    plt.ylim(-0.05, 1.05)
    plt.savefig(args.output_dir / "linguistic_2_overlap_boxplot.png", bbox_inches="tight")
    plt.close()

    # Plot 3: does word overlap relate to quality score?
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="classification", y="overlap_ratio", hue="classification", palette="Pastel1", legend=False)
    plt.title("Does Lexical Overlap Affect the Quality Score?", fontsize=14)
    plt.xlabel("Quality Score (1 = Poor, 3 = Excellent)")
    plt.ylabel("Lexical Overlap Ratio")
    plt.savefig(args.output_dir / "linguistic_3_overlap_by_score.png", bbox_inches="tight")
    plt.close()

    print(f"Saved 3 linguistic analysis plots to {args.output_dir}")


if __name__ == "__main__":
    main()
