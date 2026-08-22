"""Combine the Sarcasm Corpus V2 category files into one canonical dataset.

Reads the three raw, untouched category files under
`data/raw/sarcasm_corpus_v2/` and writes one combined table with a global
`example_id`, source metadata (`category`, `source_file`), and duplicate/
label-conflict flags. Nothing is dropped or deduplicated here -- this step
is purely normalization + metadata attachment. See PROJECT_SUMMARY.md
("Dataset Information") for why each field exists.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config.classification_settings import classification_settings

CATEGORY_FILES = {
    "GEN": "GEN-sarc-notsarc.csv",
    "HYP": "HYP-sarc-notsarc.csv",
    "RQ": "RQ-sarc-notsarc.csv",
}

LABEL_MAP = {"sarc": "sarcastic", "notsarc": "not_sarcastic"}

EXPECTED_RAW_COLUMNS = {"class", "id", "text"}


def _normalize_text(text: str) -> str:
    """Case/whitespace-insensitive key used to group duplicate posts across
    category files, so grouped splitting can keep every copy of the same
    underlying post in the same split (see make_splits.py)."""
    return " ".join(str(text).strip().lower().split())


def build_canonical_dataset(raw_dir: Path, output_path: Path | None = None) -> pd.DataFrame:
    frames = []
    for category, filename in CATEGORY_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected raw file not found: {path}")
        df = pd.read_csv(path)
        missing = EXPECTED_RAW_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing expected columns: {missing}")
        df["category"] = category
        df["source_file"] = filename
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    combined["example_id"] = combined["category"] + "-" + combined["id"].astype(str)
    if combined["example_id"].duplicated().any():
        dupes = combined.loc[combined["example_id"].duplicated(), "example_id"].tolist()
        raise ValueError(f"example_id collision after combining category files: {dupes}")

    unknown_labels = set(combined["class"].unique()) - set(LABEL_MAP)
    if unknown_labels:
        raise ValueError(f"Unexpected 'class' values in raw data: {unknown_labels}")
    combined["label"] = combined["class"].map(LABEL_MAP)

    if combined["text"].isna().any():
        raise ValueError("Found rows with missing text after combining -- inspect raw files")

    combined["normalized_text"] = combined["text"].map(_normalize_text)
    combined["dup_group_id"] = combined.groupby("normalized_text").ngroup()
    conflicted = combined.groupby("dup_group_id")["label"].nunique()
    conflict_group_ids = set(conflicted[conflicted > 1].index)
    combined["label_conflict"] = combined["dup_group_id"].isin(conflict_group_ids)

    canonical = combined[
        [
            "example_id",
            "category",
            "source_file",
            "id",
            "text",
            "label",
            "dup_group_id",
            "label_conflict",
        ]
    ].rename(columns={"id": "raw_id"})

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canonical.to_csv(output_path, index=False, encoding="utf-8-sig")

    return canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=classification_settings.sarcasm_v2_raw_dir)
    parser.add_argument("--output", type=Path, default=classification_settings.canonical_dataset_path)
    args = parser.parse_args()

    canonical = build_canonical_dataset(args.raw_dir, args.output)
    n_dup_rows = int(canonical["dup_group_id"].duplicated(keep=False).sum())
    n_conflict_rows = int(canonical["label_conflict"].sum())
    print(f"Wrote {len(canonical)} rows to {args.output}")
    print(f"  categories: {canonical['category'].value_counts().to_dict()}")
    print(f"  labels: {canonical['label'].value_counts().to_dict()}")
    print(f"  rows sharing a duplicate-text group: {n_dup_rows}")
    print(f"  rows in a label-conflict group: {n_conflict_rows}")


if __name__ == "__main__":
    main()
