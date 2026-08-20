"""Family-level grouping, deterministic sampling, and leakage guards for
SIGN's family-structured data (see `load_sign.py` for the table schema).

Every function here treats a "family" (one sarcastic original + its
interpretation rows, sharing one `family_id`) as the atomic unit -- never
split, sampled, or assigned across a boundary at the row level. This is
the single place that implements the brief's hard requirement: "never
create leakage by placing interpretations belonging to the same source
tweet across inappropriate train/evaluation boundaries."
"""
from __future__ import annotations

import random
from collections.abc import Iterable

import pandas as pd


def assert_no_family_leakage(*groups: Iterable[str], labels: list[str] | None = None) -> None:
    """Raise if any `family_id` appears in more than one of the given
    groups (each group is an iterable of family_ids, e.g. a DataFrame's
    `family_id` column). Use this after any split/sample/combine
    operation over SIGN data that is supposed to keep groups disjoint."""
    sets = [set(g) for g in groups]
    labels = labels or [f"group_{i}" for i in range(len(sets))]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise ValueError(
                    f"Family leakage between {labels[i]!r} and {labels[j]!r}: "
                    f"{len(overlap)} shared family_id(s), e.g. {sorted(overlap)[:5]}"
                )


def clean_families_only(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to families with exactly 5 interpretation rows (the
    `is_clean_family` flag from `load_sign.build_family_table`)."""
    return df[df["is_clean_family"]].copy()


def unique_family_ids(df: pd.DataFrame) -> list[str]:
    """Family IDs in the table, in stable (first-appearance) order --
    never `set(...)`, whose iteration order is not deterministic across
    runs/processes."""
    seen: dict[str, None] = {}
    for fid in df["family_id"]:
        seen.setdefault(fid, None)
    return list(seen.keys())


def sample_family_ids(
    family_ids: list[str],
    seed: int,
    frac: float | None = None,
    n: int | None = None,
) -> list[str]:
    """Deterministically sample a subset of family IDs. Exactly one of
    `frac` (0..1) or `n` (absolute count) must be given. Input is sorted
    before sampling so the result depends only on the *set* of family_ids
    and the seed, not on incidental iteration order."""
    if (frac is None) == (n is None):
        raise ValueError("Pass exactly one of frac or n")
    ordered = sorted(family_ids)
    k = n if n is not None else round(frac * len(ordered))
    k = max(0, min(k, len(ordered)))
    rng = random.Random(seed)
    return sorted(rng.sample(ordered, k))


def select_families(df: pd.DataFrame, family_ids: Iterable[str]) -> pd.DataFrame:
    """All rows (original + every interpretation) belonging to the given
    family IDs -- never a partial family."""
    wanted = set(family_ids)
    return df[df["family_id"].isin(wanted)].copy()


def select_k_interpretations_per_family(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """For every family in `df`, keep the original row plus a
    deterministic, per-family-seeded sample of `k` of its interpretation
    rows (k <= family_size). Selections are **nested**: the k=1 pick is
    contained in the k=2 pick, which is contained in the k=3 pick, etc.
    (each family's interpretation order is fixed by one seeded shuffle,
    and "k interpretations" always means "the first k of that order") --
    this is what makes the Phase 10 interpretation-count ablation a
    controlled comparison (increasing k adds interpretations rather than
    swapping in an unrelated set)."""
    keep_rows = []
    for family_id, group in df.groupby("family_id", sort=False):
        orig_rows = group[group["role"] == "original"]
        interp_rows = group[group["role"] == "interpretation"].sort_values("interp_index")
        keep_rows.append(orig_rows)

        available = interp_rows["interp_index"].tolist()
        rng = random.Random(f"{seed}:{family_id}")
        order = available[:]
        rng.shuffle(order)
        chosen = set(order[:k])
        keep_rows.append(interp_rows[interp_rows["interp_index"].isin(chosen)])

    if not keep_rows:
        return df.iloc[0:0].copy()
    return pd.concat(keep_rows, ignore_index=True)


def to_classification_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a family long-table to the minimal columns the rest of the
    project's evaluation/training code expects
    (`example_id, text, gold_label` -- `gold_label` matches the column
    name `src.classification.evaluation.metrics` requires; SIGN's
    `sarcastic`/`not_sarcastic` labels already match Dataset A's label
    set 1:1, no remapping needed)."""
    out = df[["example_id", "text", "label", "family_id", "role"]].copy()
    out = out.rename(columns={"label": "gold_label"})
    return out.reset_index(drop=True)
