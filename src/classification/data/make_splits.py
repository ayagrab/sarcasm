"""Build the one canonical train/dev/test split, reused by every approach.

Grouped by `dup_group_id` (normalized-text duplicate group) so that rows
sharing text across category files always land in the same split --
otherwise the same underlying post could appear in both train and test.
Stratified by label at the group level via `StratifiedGroupKFold`. Fixed
seed. See PROJECT_SUMMARY.md ("Decisions made") for the full rationale.

The TEST split produced here must not be touched again until final,
frozen-configuration evaluation of each approach (no prompt iteration, no
few-shot selection, no hyperparameter tuning against it).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config.classification_settings import classification_settings


def make_splits(
    df: pd.DataFrame,
    train_frac: float,
    dev_frac: float,
    test_frac: float,
    seed: int,
) -> pd.DataFrame:
    if not np.isclose(train_frac + dev_frac + test_frac, 1.0):
        raise ValueError("train/dev/test fractions must sum to 1.0")

    y = df["label"].to_numpy()
    groups = df["dup_group_id"].to_numpy()

    n_splits_test = max(round(1 / test_frac), 2)
    skf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    train_dev_idx, test_idx = next(skf_test.split(df, y, groups))

    remaining_frac = train_frac + dev_frac
    n_splits_dev = max(round(remaining_frac / dev_frac), 2)
    skf_dev = StratifiedGroupKFold(n_splits=n_splits_dev, shuffle=True, random_state=seed)
    y_rem = y[train_dev_idx]
    groups_rem = groups[train_dev_idx]
    train_rel_idx, dev_rel_idx = next(
        skf_dev.split(df.iloc[train_dev_idx], y_rem, groups_rem)
    )
    train_idx = train_dev_idx[train_rel_idx]
    dev_idx = train_dev_idx[dev_rel_idx]

    split = pd.Series(index=df.index, dtype=object)
    split.iloc[train_idx] = "train"
    split.iloc[dev_idx] = "dev"
    split.iloc[test_idx] = "test"

    assignments = df[["example_id", "dup_group_id", "category", "label"]].copy()
    assignments["split"] = split.to_numpy()

    _assert_no_group_leakage(assignments)
    return assignments


def _assert_no_group_leakage(assignments: pd.DataFrame) -> None:
    group_splits = assignments.groupby("dup_group_id")["split"].nunique()
    leaking = group_splits[group_splits > 1]
    if len(leaking):
        raise AssertionError(
            f"{len(leaking)} duplicate-text group(s) span more than one split -- leakage bug"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=classification_settings.canonical_dataset_path)
    parser.add_argument("--splits-dir", type=Path, default=classification_settings.splits_dir)
    parser.add_argument("--seed", type=int, default=classification_settings.random_seed)
    parser.add_argument("--train-frac", type=float, default=classification_settings.train_frac)
    parser.add_argument("--dev-frac", type=float, default=classification_settings.dev_frac)
    parser.add_argument("--test-frac", type=float, default=classification_settings.test_frac)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} not found -- run "
            "`python -m src.classification.data.build_canonical_dataset` first"
        )

    df = pd.read_csv(args.input)
    assignments = make_splits(df, args.train_frac, args.dev_frac, args.test_frac, args.seed)

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = args.splits_dir / "split_assignments.csv"
    assignments[["example_id", "split"]].to_csv(
        assignments_path, index=False, encoding="utf-8-sig"
    )

    merged = df.merge(assignments[["example_id", "split"]], on="example_id")
    for split_name in ("train", "dev", "test"):
        subset = merged[merged["split"] == split_name].drop(columns=["split"])
        subset.to_csv(args.splits_dir / f"{split_name}.csv", index=False, encoding="utf-8-sig")

    print(f"Wrote split assignments to {assignments_path}")
    counts = assignments["split"].value_counts()
    fracs = (counts / len(assignments) * 100).round(1)
    for name in ("train", "dev", "test"):
        print(f"  {name}: {counts.get(name, 0)} rows ({fracs.get(name, 0)}%)")

    print("\nLabel distribution per split:")
    print(merged.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("\nCategory distribution per split:")
    print(merged.groupby(["split", "category"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
