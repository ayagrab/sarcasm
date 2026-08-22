"""Reusable validation/audit script for the canonical sarcasm dataset.

Run after `build_canonical_dataset.py`. Computes and *reports* (never
silently fixes or drops) data-quality signals: class balance, missing
values, duplicate/near-duplicate text, label conflicts, and length
distribution. Intended to be re-run any time the canonical dataset changes,
so drift is caught immediately rather than discovered mid-experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config.classification_settings import classification_settings


def audit_dataset(df: pd.DataFrame) -> dict:
    report: dict = {}

    report["n_rows"] = int(len(df))
    report["n_missing_text"] = int(df["text"].isna().sum())
    report["n_missing_label"] = int(df["label"].isna().sum())
    report["n_example_id_duplicates"] = int(df["example_id"].duplicated().sum())

    report["label_counts"] = {k: int(v) for k, v in df["label"].value_counts().items()}
    report["category_counts"] = {k: int(v) for k, v in df["category"].value_counts().items()}

    cat_label = df.groupby(["category", "label"]).size().unstack(fill_value=0)
    report["category_label_counts"] = {
        str(cat): {str(lab): int(n) for lab, n in row.items()}
        for cat, row in cat_label.iterrows()
    }

    dup_rows = df[df.duplicated("dup_group_id", keep=False)]
    report["n_rows_in_duplicate_text_group"] = int(len(dup_rows))
    report["n_duplicate_text_groups"] = int(dup_rows["dup_group_id"].nunique())
    report["n_rows_label_conflict"] = int(df["label_conflict"].sum())
    report["label_conflict_example_ids"] = df.loc[df["label_conflict"], "example_id"].tolist()

    word_lens = df["text"].dropna().str.split().apply(len)
    report["word_length"] = {
        "min": int(word_lens.min()),
        "p25": float(word_lens.quantile(0.25)),
        "median": float(word_lens.median()),
        "mean": float(word_lens.mean()),
        "p75": float(word_lens.quantile(0.75)),
        "max": int(word_lens.max()),
    }
    report["n_very_short_le2_words"] = int((word_lens <= 2).sum())
    report["n_very_long_gt200_words"] = int((word_lens > 200).sum())

    report["warnings"] = _build_warnings(report)
    return report


def _build_warnings(report: dict) -> list[str]:
    warnings = []
    if report["n_missing_text"]:
        warnings.append(f"{report['n_missing_text']} rows have missing text")
    if report["n_missing_label"]:
        warnings.append(f"{report['n_missing_label']} rows have missing label")
    if report["n_example_id_duplicates"]:
        warnings.append(f"{report['n_example_id_duplicates']} duplicate example_id values")
    if report["n_rows_label_conflict"]:
        warnings.append(
            f"{report['n_rows_label_conflict']} rows belong to a duplicate-text group with "
            "conflicting labels -- kept, not dropped (see PROJECT_SUMMARY.md)"
        )
    labels = report["label_counts"]
    if len(labels) == 2:
        counts = list(labels.values())
        imbalance = max(counts) / max(min(counts), 1)
        if imbalance > 1.2:
            warnings.append(f"class imbalance ratio {imbalance:.2f} exceeds 1.2")
    if report["n_very_short_le2_words"]:
        warnings.append(f"{report['n_very_short_le2_words']} rows have <=2 words")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=classification_settings.canonical_dataset_path)
    parser.add_argument("--output", type=Path, default=classification_settings.audit_report_path)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} not found -- run "
            "`python -m src.classification.data.build_canonical_dataset` first"
        )
    df = pd.read_csv(args.input)
    report = audit_dataset(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote audit report to {args.output}")
    if report["warnings"]:
        print(f"\n{len(report['warnings'])} warning(s) -- see 'warnings' key above")


if __name__ == "__main__":
    main()
