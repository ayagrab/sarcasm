"""Boxplots of BLEU/ROUGE/PINC, by model and by prompt.

Reads the summary produced by `summarize_text_metrics.py`. Reproduces the
NLP-metrics figures referenced in docs/meeting_notes_summary.md (2026-07-16
meeting): as prompts progress 1->4, BLEU/ROUGE rise and PINC falls; Liquid's
extreme PINC / near-zero BLEU explains its low human quality scores.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import settings
from src.common.file_utils import ensure_parent_dir

METRICS = ["BLEU", "ROUGE-1", "ROUGE-2", "PINC"]


def plot_by(df: pd.DataFrame, group_column: str, output_path: Path, title: str) -> None:
    melted = df.melt(id_vars=["model", "prompt"], value_vars=METRICS, var_name="Metric", value_name="Score")
    g = sns.catplot(
        data=melted, x=group_column, y="Score", hue=group_column, col="Metric", col_wrap=2,
        kind="box", sharey=False, height=4, aspect=1.2, palette="Set2", legend=False,
    )
    g.figure.suptitle(title, y=1.05, fontsize=16)
    ensure_parent_dir(output_path)
    g.savefig(output_path, bbox_inches="tight")
    plt.close(g.figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BLEU/ROUGE/PINC distributions by model and by prompt.")
    parser.add_argument("--input", type=Path, default=settings.summaries_dir / "text_metrics_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=settings.summaries_dir / "figures")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_by(df, "model", args.output_dir / "avg_all_models_per_score.png", "Score Distribution by Model (Across Prompts)")
    plot_by(df, "prompt", args.output_dir / "avg_all_prompts_per_score.png", "Score Distribution by Prompt (Across Models)")
    print(f"Saved 2 plots to {args.output_dir}")


if __name__ == "__main__":
    main()
