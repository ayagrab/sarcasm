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
    """DEPRECATED for the primary-reference-based experiments (Phase
    7/9/10 as of 2026-08-20) -- kept only as a documented alternative,
    NOT used by default anymore. For every family in `df`, keep the
    original row plus a deterministic, per-family-seeded *shuffled*
    sample of `k` of its interpretation rows (k <= family_size).
    Selections are nested (k=1 subset of k=2 subset of k=3, ...) but
    which interpretation counts as "first" is randomized per family, not
    rank-based. Superseded by `select_top_k_interpretations_per_family`,
    which uses interpretation #1 (the primary/best human reference, see
    `load_sign.py`'s docstring) as the anchor instead of a random pick --
    use that one unless there is a specific reason to want the
    shuffled variant."""
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


def select_top_k_interpretations_per_family(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """For every family in `df`, keep the original row plus
    interpretations #1..k **by rank** (`interp_index` 1..k, i.e. the
    primary/best reference first, then #2, #3, ... in the fixed order
    already recorded in the family table -- never shuffled, no seed
    needed since there is nothing random to control). Families with
    `family_size < k` contribute only their available interpretations
    (documented, not padded/invented -- see `PROJECT_SUMMARY.md`
    section 1 on anomalous/incomplete families).

    This is the primary selection function for Phase 7 (SIGN Train
    variants), Phase 9 (learning curve), and Phase 10 (interpretation
    -count ablation) as of the 2026-08-20 "interpretation #1 is primary"
    clarification -- k=1 gives exactly the "original + interpretation #1"
    balanced condition; k=1/2/3/5 are nested by construction (k=1's
    interpretation is contained in k=2's, etc.), since they're just
    prefixes of the same fixed rank order."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    keep_rows = []
    for _, group in df.groupby("family_id", sort=False):
        orig_rows = group[group["role"] == "original"]
        interp_rows = group[(group["role"] == "interpretation") & (group["interp_index"] <= k)]
        keep_rows.append(orig_rows)
        keep_rows.append(interp_rows)
    if not keep_rows:
        return df.iloc[0:0].copy()
    return pd.concat(keep_rows, ignore_index=True)


def select_primary_interpretation_per_family(df: pd.DataFrame) -> pd.DataFrame:
    """The primary balanced SIGN condition: original + interpretation #1
    only, for every family in `df`. Equivalent to
    `select_top_k_interpretations_per_family(df, k=1)`, kept as a
    separate named entry point since it's the default/main condition
    referenced throughout Phase 7-10, not just one point on the k-sweep."""
    return select_top_k_interpretations_per_family(df, k=1)


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
