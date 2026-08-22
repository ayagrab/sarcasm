"""Parse SIGN's raw train/dev/test files into a family-aware long table.

The raw files (`data/raw/original_{split}_dataset.csv`) have no header and
no tweet-ID column -- each row is one `(sarcastic_original,
human_interpretation)` pair, five rows per source tweet ("family"). Exact
stripped original text is the only key the data supports for grouping
rows back into families; see `PROJECT_SUMMARY.md` section 1 for
the full audit (counts, and why grouping by text yields fewer families
than the paper's official 2,400/300/300, plus a handful of families that
don't have exactly 5 interpretation rows).

Output schema (one row per original OR per interpretation -- long format):

    family_id      str   stable within a split, e.g. "train-00001"
    split          str   "train" | "dev" | "test"
    role           str   "original" | "interpretation"
    interp_index   int   0 for the original; 1..N for interpretations, in
                          the exact order they appear for that family in
                          the raw source file (N == family_size; usually
                          5, sometimes not -- see `is_clean_family`).
                          **This doubles as interpretation RANK**:
                          interp_index == 1 is treated as "interpretation
                          #1", the primary/best human reference for that
                          family (per project decision, 2026-08-20) --
                          used wherever a single canonical non-sarcastic
                          reference is needed (Phase 5's primary-reference
                          view, Phase 7's primary balanced training
                          condition, Phase 9's learning curve, Phase 10's
                          nested k=1..5 interpretation-count ablation).
                          CAVEAT, disclosed rather than assumed silently:
                          the raw files carry no interpretation-ID column,
                          so "rank" here means "row order as already
                          present in the committed raw file" -- there is
                          no independent way, from the data available in
                          this repo, to confirm that order matches
                          whatever numbering the original SIGN release
                          used internally. Treated as the best available
                          proxy, not a verified ground truth.
    is_primary_interpretation bool True iff role=="interpretation" and
                          interp_index == 1 (convenience flag, exactly
                          equivalent to that condition -- never used to
                          derive anything, just avoids repeating the
                          `interp_index == 1` filter everywhere).
    text           str
    label          str   "sarcastic" (role=original) | "not_sarcastic" (role=interpretation)
    example_id     str   f"{family_id}-orig" | f"{family_id}-interp{interp_index}"
    family_size    int   number of interpretation rows in this family
    is_clean_family bool True iff family_size == 5 (see plan doc for why
                          this isn't always true)
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path

import pandas as pd

from config.sign_settings import sign_settings

FAMILY_TABLE_COLUMNS = [
    "family_id",
    "split",
    "role",
    "interp_index",
    "is_primary_interpretation",
    "text",
    "label",
    "example_id",
    "family_size",
    "is_clean_family",
]


def load_raw_pairs(split: str, path: Path | None = None) -> list[tuple[str, str]]:
    """Read one split's raw (original, interpretation) pairs, stripped.

    Uses the stdlib `csv` module, not `pandas.read_csv` -- the raw files
    have no header, so `pandas` would silently treat the first data row
    as a header unless told otherwise; `csv.reader` avoids that class of
    mistake entirely. Every row is verified to have exactly 2 fields
    (raises otherwise) -- confirmed true for all 3 splits during the
    Phase 0 audit, kept as a guard against a corrupted/edited file.

    `path` overrides the configured raw-file location for `split` --
    used by tests to exercise this function against small synthetic
    fixtures instead of the real (thousand-row) SIGN files.
    """
    if path is None:
        raw_files = dict(sign_settings.raw_files)
        if split not in raw_files:
            raise ValueError(f"Unknown SIGN split {split!r}; expected one of {list(raw_files)}")
        path = raw_files[split]
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if len(row) != 2:
                raise ValueError(f"{path}:{i + 1}: expected 2 fields, got {len(row)}: {row!r}")
            orig, interp = row
            pairs.append((orig.strip(), interp.strip()))
    return pairs


def build_family_table(split: str, path: Path | None = None) -> pd.DataFrame:
    """Group one split's raw pairs into families (by exact original text,
    in order of first appearance) and return the long-format table
    described in this module's docstring. `path` -- see `load_raw_pairs`."""
    pairs = load_raw_pairs(split, path=path)

    families: "OrderedDict[str, list[str]]" = OrderedDict()
    for orig, interp in pairs:
        families.setdefault(orig, []).append(interp)

    rows = []
    for i, (orig_text, interps) in enumerate(families.items(), start=1):
        family_id = f"{split}-{i:05d}"
        family_size = len(interps)
        is_clean = family_size == 5
        rows.append(
            {
                "family_id": family_id,
                "split": split,
                "role": "original",
                "interp_index": 0,
                "is_primary_interpretation": False,
                "text": orig_text,
                "label": sign_settings.original_label,
                "example_id": f"{family_id}-orig",
                "family_size": family_size,
                "is_clean_family": is_clean,
            }
        )
        for j, interp_text in enumerate(interps, start=1):
            rows.append(
                {
                    "family_id": family_id,
                    "split": split,
                    "role": "interpretation",
                    "interp_index": j,
                    "is_primary_interpretation": j == 1,
                    "text": interp_text,
                    "label": sign_settings.interpretation_label,
                    "example_id": f"{family_id}-interp{j}",
                    "family_size": family_size,
                    "is_clean_family": is_clean,
                }
            )
    return pd.DataFrame(rows, columns=FAMILY_TABLE_COLUMNS)


def family_table_path(split: str) -> Path:
    return sign_settings.processed_dir / f"family_table_{split}.csv"


def save_family_table(split: str) -> pd.DataFrame:
    df = build_family_table(split)
    out_path = family_table_path(split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def load_family_table(split: str, rebuild: bool = False) -> pd.DataFrame:
    """Load the persisted family table, building it first if it doesn't
    exist yet (or if `rebuild=True`)."""
    path = family_table_path(split)
    if rebuild or not path.exists():
        return save_family_table(split)
    return pd.read_csv(path, encoding="utf-8-sig")


def summarize(df: pd.DataFrame) -> dict:
    """Family-level summary stats for one split's table -- the same
    numbers audited by hand in `PROJECT_SUMMARY.md` section 1,
    computed here so they can be asserted against in tests instead of
    only living in a one-off audit script."""
    originals = df[df["role"] == "original"]
    interps = df[df["role"] == "interpretation"]
    n_families = len(originals)
    n_clean = int(originals["is_clean_family"].sum())
    return {
        "n_rows": len(df),
        "n_originals": n_families,
        "n_interpretations": len(interps),
        "n_families": n_families,
        "n_clean_families": n_clean,
        "n_anomalous_families": n_families - n_clean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SIGN family tables for all splits.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if cached CSVs exist.")
    args = parser.parse_args()

    for split, _ in sign_settings.raw_files:
        df = load_family_table(split, rebuild=args.rebuild)
        stats = summarize(df)
        print(f"[{split}] {stats}")
        print(f"  -> saved to {family_table_path(split)}")


if __name__ == "__main__":
    main()
