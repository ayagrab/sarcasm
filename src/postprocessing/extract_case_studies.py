"""Extract concrete examples where human and LLM-judge scores agree or clash.

Builds on the same human/LLM comparison table as `human_llm_agreement.py`,
with the model's actual interpretation text attached for qualitative
review. Reproduces the case-study extraction described in
docs/meeting_notes_summary.md (2026-07-16 meeting): the "semantic rigidity"
and "fluency bias" examples came from reviewing exactly these discrepancy
cases.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from config.settings import settings
from src.common.file_utils import save_csv
from src.postprocessing.human_llm_agreement import attach_interpretations, build_comparison_table

PREVIEW_COLUMNS = ["sentence", "model_interpretation", "human_score", "chatgpt_score", "model", "folder"]


def find_perfect_agreement_cases(merged):
    """Rows where the rounded human score exactly matches the ChatGPT score."""
    return merged[merged["human_score_rounded"] == merged["chatgpt_score"]].copy()


def find_discrepancy_cases(merged, top_n: int = 10):
    """The `top_n` rows with the largest gap between human and ChatGPT scores."""
    merged = merged.copy()
    merged["evaluator_diff"] = (merged["human_score"] - merged["chatgpt_score"]).abs()
    return merged.sort_values("evaluator_diff", ascending=False).head(top_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract human/LLM agreement and discrepancy case studies.")
    parser.add_argument("--alt-test-dir", type=Path, default=settings.data_dir / "alt_test")
    parser.add_argument("--outputs-dir", type=Path, default=settings.model_outputs_dir)
    parser.add_argument("--summaries-dir", type=Path, default=settings.summaries_dir)
    parser.add_argument("--top-n-discrepancies", type=int, default=10)
    args = parser.parse_args()

    merged = build_comparison_table(args.alt_test_dir / "humans_annotations.json", args.alt_test_dir / "llm_annotations.json")
    merged = attach_interpretations(merged, args.outputs_dir)

    agreement_df = find_perfect_agreement_cases(merged)
    rate = len(agreement_df) / len(merged) * 100
    print(f"Perfect agreement: {len(agreement_df)}/{len(merged)} instances ({rate:.1f}%).")
    agreement_path = args.summaries_dir / "perfect_agreement_cases.csv"
    save_csv(agreement_df[PREVIEW_COLUMNS], agreement_path)
    print(f"Saved to {agreement_path}")

    discrepancy_df = find_discrepancy_cases(merged, args.top_n_discrepancies)
    discrepancy_path = args.summaries_dir / "discrepancy_cases_for_discussion.csv"
    save_csv(discrepancy_df[PREVIEW_COLUMNS], discrepancy_path)
    print(f"\nTop {args.top_n_discrepancies} discrepancies saved to {discrepancy_path}")
    print(discrepancy_df[PREVIEW_COLUMNS].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
