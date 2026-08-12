"""Few-shot demonstration selection for LLM prompting.

Selection only ever draws from TRAIN (never dev/test), and is deterministic
given (variant, n_shots, seed) so every evaluated example sees the exact
same demonstration set within one experiment. Selection never looks at
labels/performance on dev or test -- only at the (label, category)
composition of TRAIN itself. Callers must record the returned example IDs
in the experiment config (task Section 8: "Record exactly which example
IDs were included in every prompt").
"""
from __future__ import annotations

import pandas as pd


def select_random_few_shot(train_df: pd.DataFrame, n_shots: int, seed: int) -> pd.DataFrame:
    """Random selection, balanced 50/50 across labels."""
    labels = sorted(train_df["label"].unique())
    per_label = n_shots // len(labels)
    remainder = n_shots - per_label * len(labels)
    parts = []
    for i, label in enumerate(labels):
        take = per_label + (1 if i < remainder else 0)
        pool = train_df[train_df["label"] == label]
        parts.append(pool.sample(n=min(take, len(pool)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def select_curated_few_shot(train_df: pd.DataFrame, n_shots: int, seed: int) -> pd.DataFrame:
    """Diverse selection: stratified across (category, label) cells, so
    demonstrations cover different sarcasm sub-phenomena (GEN/HYP/RQ) and
    both labels, rather than a random slice dominated by the majority
    category (GEN is ~70% of the data)."""
    cells = list(train_df.groupby(["category", "label"], sort=True))
    per_cell = max(n_shots // len(cells), 1)

    picked_parts = []
    for _, group in cells:
        picked_parts.append(group.sample(n=min(per_cell, len(group)), random_state=seed))
    picked = pd.concat(picked_parts)

    if len(picked) > n_shots:
        picked = picked.sample(n=n_shots, random_state=seed)
    elif len(picked) < n_shots:
        pool = train_df[~train_df["example_id"].isin(picked["example_id"])]
        extra = pool.sample(n=min(n_shots - len(picked), len(pool)), random_state=seed)
        picked = pd.concat([picked, extra])

    return picked.sample(frac=1, random_state=seed).reset_index(drop=True)
