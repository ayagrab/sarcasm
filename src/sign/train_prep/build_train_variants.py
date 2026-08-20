"""Phase 7: prepare SIGN Train variants (data prep only, no training yet
-- SIGN_GENERALIZATION_PLAN.md, Phase 7). Gated on Phase 4/5/6 being
persisted first (brief's hard requirement) -- this is the first phase
that touches SIGN Train at all.

Produces one CSV per condition under `data/sign/train_variants/`:
  - `primary.csv`  -- original + interpretation #1 only (the default
    condition for Phase 8/9's domain adaptation and learning curve).
  - `k2.csv`, `k3.csv`, `k5.csv` -- original + interpretations #1..k by
    rank (Phase 10's interpretation-count ablation), nested by construction.

Each variant keeps the family table's native `text`/`label` columns (no
renaming needed -- both M1's `tfidf_baseline.build_pipeline` and M6's
`finetune.build_dataset` already train on exactly these two columns), plus
`example_id`/`family_id`/`role`/`interp_index` for traceability. A JSON
metadata sidecar per variant records exact family/example counts and,
for k>1, the *documented* (not pre-applied) imbalance-handling policy --
`class_weight="balanced"` for M1, duplicate-the-original-k-times for M6
-- to be applied by whichever Phase 8/10 training script consumes it.

Known limitation, explicitly kept rather than silently handled (2026-08-20
user decision after Phase 6's error analysis): ~25% of Train families
have interpretation #1 byte-identical to their own original (see
SIGN_GENERALIZATION_PLAN.md §1). These families are NOT filtered out or
special-cased here -- the primary condition trains on them as-is, which
means a real fraction of "sarcastic" and "not_sarcastic" training pairs
are the identical input string. This is documented, not fixed, per the
user's explicit choice to keep the natural condition and treat it as a
disclosed data limitation.
"""
from __future__ import annotations

import json

import pandas as pd

from config.sign_settings import sign_settings
from src.sign.data.family_utils import (
    assert_no_family_leakage,
    select_primary_interpretation_per_family,
    select_top_k_interpretations_per_family,
    unique_family_ids,
)
from src.sign.data.load_sign import load_family_table

VARIANT_COLUMNS = ["example_id", "family_id", "role", "interp_index", "text", "label"]

IMBALANCE_POLICY = {
    "M1_tfidf_logreg": 'class_weight="balanced" in LogisticRegression (documented, applied at training time by Phase 8/10, not here)',
    "M6_deberta": "duplicate the sarcastic original k times so the training set stays 1:1 (documented, applied at training time by Phase 8/10, not here)",
}

DUPLICATE_INTERP1_LIMITATION = (
    "~25% of Train families have interpretation #1 byte-identical to their own "
    "original (SIGN_GENERALIZATION_PLAN.md §1, discovered in Phase 6). Kept as-is "
    "per explicit 2026-08-20 user decision -- not filtered, not fixed. Every variant "
    "below (including k>1) inherits this property for whichever ranks it includes."
)


def _variant_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[VARIANT_COLUMNS].reset_index(drop=True)


def build_primary_variant(train_table: pd.DataFrame) -> pd.DataFrame:
    return _variant_frame(select_primary_interpretation_per_family(train_table))


def build_k_variant(train_table: pd.DataFrame, k: int) -> pd.DataFrame:
    return _variant_frame(select_top_k_interpretations_per_family(train_table, k))


def verify_no_cross_split_leakage(variant: pd.DataFrame, dev_table: pd.DataFrame, test_table: pd.DataFrame) -> None:
    assert_no_family_leakage(
        unique_family_ids(variant),
        unique_family_ids(dev_table),
        unique_family_ids(test_table),
        labels=["variant", "dev", "test"],
    )


def variant_metadata(name: str, variant: pd.DataFrame, imbalance_policy: str | None) -> dict:
    n_originals = int((variant["role"] == "original").sum())
    n_interps = int((variant["role"] == "interpretation").sum())
    return {
        "variant": name,
        "n_families": int(variant["family_id"].nunique()),
        "n_originals": n_originals,
        "n_interpretations": n_interps,
        "n_total_rows": int(len(variant)),
        "interp_ranks_included": sorted(int(r) for r in variant.loc[variant["role"] == "interpretation", "interp_index"].unique()),
        "class_balance": {"sarcastic": n_originals, "not_sarcastic": n_interps},
        "imbalance_handling_policy": imbalance_policy,
        "duplicate_interpretation1_limitation": DUPLICATE_INTERP1_LIMITATION,
    }


def main() -> None:
    train_table = load_family_table("train")
    dev_table = load_family_table("dev")
    test_table = load_family_table("test")

    out_dir = sign_settings.processed_dir / "train_variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants: dict[str, tuple[pd.DataFrame, str | None]] = {
        "primary": (build_primary_variant(train_table), None),
        "k2": (build_k_variant(train_table, 2), json.dumps(IMBALANCE_POLICY)),
        "k3": (build_k_variant(train_table, 3), json.dumps(IMBALANCE_POLICY)),
        "k5": (build_k_variant(train_table, 5), json.dumps(IMBALANCE_POLICY)),
    }

    for name, (variant, policy) in variants.items():
        verify_no_cross_split_leakage(variant, dev_table, test_table)
        variant.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        meta = variant_metadata(name, variant, policy)
        with open(out_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(
            f"{name}.csv: {meta['n_families']} families, {meta['n_originals']} sarcastic + "
            f"{meta['n_interpretations']} not_sarcastic (ranks {meta['interp_ranks_included']})"
        )


if __name__ == "__main__":
    main()
