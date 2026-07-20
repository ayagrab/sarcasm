"""Kruskal-Wallis significance tests: does prompt or model choice affect quality?

Reproduces the project's documented result (see docs/meeting_notes_summary.md,
2026-07-16 meeting): prompt effect p=9.45e-34, model effect p=5.13e-72 --
both far below 0.05, i.e. both prompt and model choice have a statistically
decisive impact on the LLM judge's classification score.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from scipy import stats
from config.settings import settings
from src.common.file_utils import load_all_model_outputs


def kruskal_wallis_by(df, group_column: str) -> tuple[float, float]:
    """Run Kruskal-Wallis on `classification`, grouped by `group_column`."""
    groups = [group["classification"].values for _, group in df.groupby(group_column)]
    return stats.kruskal(*groups)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kruskal-Wallis tests for prompt and model effects on quality score.")
    parser.add_argument("--outputs-dir", type=Path, default=settings.model_outputs_dir)
    args = parser.parse_args()

    df = load_all_model_outputs(args.outputs_dir)
    print(f"Loaded {len(df)} classified rows across {df['prompt'].nunique()} prompts and {df['model'].nunique()} models.")

    stat_prompt, p_prompt = kruskal_wallis_by(df, "prompt")
    print("\n--- Prompt effect ---")
    print(f"Kruskal-Wallis Statistic: {stat_prompt:.3f}")
    print(f"p-value: {p_prompt:.4e}")
    print("Significant difference between prompts." if p_prompt < 0.05 else "No significant difference between prompts.")

    stat_model, p_model = kruskal_wallis_by(df, "model")
    print("\n--- Model effect ---")
    print(f"Kruskal-Wallis Statistic: {stat_model:.3f}")
    print(f"p-value: {p_model:.4e}")
    print("Significant difference between models." if p_model < 0.05 else "No significant difference between models.")


if __name__ == "__main__":
    main()
