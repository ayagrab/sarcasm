"""Bulk BLEU/ROUGE/PINC summary across every model-output file.

Runs `calculate_text_metrics.calculate_metrics` (the same per-file metric
calculation used elsewhere in the project) over every file under
data/model_outputs/, instead of one file at a time.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from config.settings import settings
from src.common.file_utils import save_csv
from src.postprocessing.calculate_text_metrics import calculate_metrics


def summarize_all(outputs_dir: Path) -> pd.DataFrame:
    """Compute BLEU/ROUGE/PINC for every model-output CSV, with prompt/model columns."""
    rows = []
    for csv_file in sorted(outputs_dir.glob("experiment_*/*.csv")):
        summary = calculate_metrics(csv_file).iloc[0]
        rows.append({
            "prompt": int(csv_file.parent.name.split("_")[-1]),
            "model": csv_file.stem.split("_")[0],
            "BLEU": summary["avg_bleu"] * 100,
            "ROUGE-1": summary["avg_rouge1_recall"] * 100,
            "ROUGE-2": summary["avg_rouge2_recall"] * 100,
            "PINC": summary["avg_pinc"] * 100,
        })
    return pd.DataFrame(rows).sort_values(["prompt", "model"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize BLEU/ROUGE/PINC across every model-output file.")
    parser.add_argument("--outputs-dir", type=Path, default=settings.model_outputs_dir)
    parser.add_argument("--output", type=Path, default=settings.summaries_dir / "text_metrics_summary.csv")
    args = parser.parse_args()

    summary_df = summarize_all(args.outputs_dir)
    save_csv(summary_df, args.output)
    print(summary_df.to_string(index=False))
    print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
