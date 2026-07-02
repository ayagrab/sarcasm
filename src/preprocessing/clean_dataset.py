"""Clean the original sarcasm dataset.

The original test CSV contains multiple fields. This script extracts the
sarcastic sentence from the first column, removes duplicates, and saves a clean
single-column dataset.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from config.settings import settings
from src.common.file_utils import save_csv


def clean_sarcastic_sentences(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Extract unique sarcastic sentences from the first CSV column."""
    raw_df = pd.read_csv(input_path, header=None, encoding="utf-8-sig")
    sentences = (
        raw_df.iloc[:, 0]
        .dropna()
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
    )
    clean_df = pd.DataFrame({"sarcastic_sentence": sorted(set(sentences))})
    save_csv(clean_df, output_path)
    return clean_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the raw sarcasm dataset.")
    parser.add_argument("--input", type=Path, default=settings.raw_data_dir / "original_test_dataset.csv")
    parser.add_argument("--output", type=Path, default=settings.processed_data_dir / "clean_sarcastic_sentences.csv")
    args = parser.parse_args()
    clean_df = clean_sarcastic_sentences(args.input, args.output)
    print(f"Saved {len(clean_df)} unique sentences to {args.output}")


if __name__ == "__main__":
    main()
