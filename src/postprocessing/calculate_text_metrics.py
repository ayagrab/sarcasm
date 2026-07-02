"""Calculate automatic text-overlap and novelty metrics.

Metrics:
- BLEU: precision-oriented overlap.
- ROUGE-1 / ROUGE-2: recall-oriented overlap.
- PINC: novelty, i.e., how many interpretation words do not appear in the original.
- Combined: PINC * sigmoid(BLEU), following the original experiment script.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
try:
    from rouge_score import rouge_scorer
except ModuleNotFoundError:  # Allows the script to run even before optional ROUGE package installation.
    rouge_scorer = None
from config.settings import settings
from src.common.file_utils import read_csv_flexible, save_csv


def calculate_pinc(original: str, interpretation: str) -> float:
    """Calculate word-level novelty as a decimal in the range [0, 1]."""
    original_words = set(str(original).lower().split())
    interpretation_words = str(interpretation).lower().split()
    if not interpretation_words:
        return 0.0
    new_words = [word for word in interpretation_words if word not in original_words]
    return len(new_words) / len(interpretation_words)



def simple_ngram_recall(reference_text: str, candidate_text: str, n: int) -> float:
    """Small fallback for ROUGE-N recall when rouge-score is not installed."""
    reference_tokens = str(reference_text).lower().split()
    candidate_tokens = str(candidate_text).lower().split()
    if len(reference_tokens) < n:
        return 0.0
    reference_ngrams = [tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens)-n+1)]
    candidate_ngrams = set(tuple(candidate_tokens[i:i+n]) for i in range(max(0, len(candidate_tokens)-n+1)))
    if not reference_ngrams:
        return 0.0
    overlap = sum(1 for gram in reference_ngrams if gram in candidate_ngrams)
    return overlap / len(reference_ngrams)

def sigmoid(value: float) -> float:
    """Standard sigmoid function."""
    return float(1 / (1 + np.exp(-value)))


def calculate_metrics(input_file: Path, output_file: Path | None = None) -> pd.DataFrame:
    """Calculate BLEU, ROUGE and PINC for a model-output CSV."""
    df = read_csv_flexible(input_file, ["sarcastic_sentence", "model_interpretation"])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True) if rouge_scorer else None
    smooth = SmoothingFunction().method1
    rows = []
    for _, row in df.iterrows():
        original = str(row["sarcastic_sentence"])
        interpretation = str(row["model_interpretation"])
        reference = [original.lower().split()]
        candidate = interpretation.lower().split()
        bleu = sentence_bleu(reference, candidate, smoothing_function=smooth)
        if scorer:
            rouge_scores = scorer.score(original, interpretation)
            rouge1_recall = rouge_scores["rouge1"].recall
            rouge2_recall = rouge_scores["rouge2"].recall
        else:
            rouge1_recall = simple_ngram_recall(original, interpretation, n=1)
            rouge2_recall = simple_ngram_recall(original, interpretation, n=2)
        pinc = calculate_pinc(original, interpretation)
        rows.append({
            "bleu": bleu,
            "rouge1_recall": rouge1_recall,
            "rouge2_recall": rouge2_recall,
            "pinc": pinc,
            "combined_pinc_sigmoid_bleu": pinc * sigmoid(bleu),
        })
    metrics_df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "file": input_file.name,
        "avg_bleu": metrics_df["bleu"].mean(),
        "avg_rouge1_recall": metrics_df["rouge1_recall"].mean(),
        "avg_rouge2_recall": metrics_df["rouge2_recall"].mean(),
        "avg_pinc": metrics_df["pinc"].mean(),
        "avg_combined_pinc_sigmoid_bleu": metrics_df["combined_pinc_sigmoid_bleu"].mean(),
        "rows": len(metrics_df),
    }])
    if output_file:
        save_csv(summary, output_file)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate BLEU, ROUGE, PINC and combined metrics.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=settings.summaries_dir / "text_metrics_summary.csv")
    args = parser.parse_args()
    summary = calculate_metrics(args.input, args.output)
    print(summary.to_string(index=False))
    print(f"Saved metrics summary to {args.output}")

if __name__ == "__main__":
    main()
