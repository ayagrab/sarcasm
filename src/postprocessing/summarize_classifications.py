"""Summarize 1/2/3 classification scores for every model-output CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from config.settings import settings
from src.common.file_utils import save_csv


def infer_model_name(csv_path: Path) -> str:
    """Infer a readable model name from a normalized output filename."""
    return csv_path.stem.replace("_run_", " run ")


def summarize_outputs(outputs_dir: Path, output_file: Path) -> pd.DataFrame:
    """Create a summary table of score averages and counts."""
    rows: list[dict] = []
    for csv_file in sorted(outputs_dir.glob("**/*.csv")):
        df = pd.read_csv(csv_file, encoding="utf-8-sig")
        if "classification" not in df.columns:
            print(f"Skipping {csv_file}: no classification column")
            continue
        scores = pd.to_numeric(df["classification"], errors="coerce").dropna()
        if scores.empty:
            print(f"Skipping {csv_file}: no classified rows")
            continue
        rows.append({
            "experiment": csv_file.parent.name,
            "model": infer_model_name(csv_file),
            "average": round(float(scores.mean()), 3),
            "median": float(scores.median()),
            "count_1": int((scores == 1).sum()),
            "count_2": int((scores == 2).sum()),
            "count_3": int((scores == 3).sum()),
            "total": int(len(scores)),
        })
    summary_df = pd.DataFrame(rows).sort_values(["experiment", "model"]).reset_index(drop=True)
    save_csv(summary_df, output_file)
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize classification scores.")
    parser.add_argument("--outputs-dir", type=Path, default=settings.model_outputs_dir)
    parser.add_argument("--output", type=Path, default=settings.summaries_dir / "classification_summary.csv")
    args = parser.parse_args()
    summary_df = summarize_outputs(args.outputs_dir, args.output)
    print(summary_df)
    print(f"Saved summary to {args.output}")

if __name__ == "__main__":
    main()
