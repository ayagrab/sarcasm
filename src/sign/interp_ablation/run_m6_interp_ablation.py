"""Phase 10, M6 leg: interpretation-count ablation (PROJECT_SUMMARY.md,
Phase 10). RQ4. k=1 (Phase 8's condition B, EXP-SIGN-023) is reused, not
rerun. k=2/3/5 are new: Dataset A TRAIN (full, fixed) + Phase 7's k2/k3/k5
variants, with the sarcastic-original rows **duplicated k times** within
the SIGN portion so the SIGN-side class balance stays 1:1 for every k
(documented policy, Phase 7's `.meta.json` sidecars) -- simpler and more
reproducible than a custom weighted loss, and isolates interpretation
*diversity* from a class-imbalance confound. Same full 100% family set at
every k (not swept -- that's Phase 9).

Reuses `run_m6_domain_adaptation.run_condition` directly. VM required.
Run: `python -m src.sign.interp_ablation.run_m6_interp_ablation`
"""
from __future__ import annotations

import pandas as pd

from config.classification_settings import classification_settings
from config.sign_settings import sign_settings
from src.sign.domain_adaptation.run_m6_domain_adaptation import run_condition

K_VALUES_TO_RUN = {2: "EXP-SIGN-036", 3: "EXP-SIGN-037", 5: "EXP-SIGN-038"}


def duplicate_originals_for_balance(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Within one k-variant's SIGN rows (1 original + k interpretations
    per family), duplicate the original rows k times so the SIGN portion
    itself is 1:1 balanced -- documented policy, not silent."""
    originals = df[df["role"] == "original"]
    interps = df[df["role"] == "interpretation"]
    duplicated = []
    for i in range(k):
        copy = originals.copy()
        copy["example_id"] = copy["example_id"] + f"-dup{i}"
        duplicated.append(copy)
    return pd.concat(duplicated + [interps], ignore_index=True)


def main() -> None:
    dataset_a_train = pd.read_csv(classification_settings.splits_dir / "train.csv", encoding="utf-8-sig")
    dataset_a_dev = pd.read_csv(classification_settings.splits_dir / "dev.csv", encoding="utf-8-sig")

    for k, experiment_id in K_VALUES_TO_RUN.items():
        sign_k = pd.read_csv(sign_settings.processed_dir / "train_variants" / f"k{k}.csv", encoding="utf-8-sig")
        sign_k_balanced = duplicate_originals_for_balance(sign_k, k)
        combined_train = pd.concat(
            [
                dataset_a_train[["example_id", "text", "label"]],
                sign_k_balanced[["example_id", "text", "label"]],
            ],
            ignore_index=True,
        )
        print(
            f"k={k}: {experiment_id} -- {len(combined_train)} train rows "
            f"({len(sign_k_balanced)} from SIGN, balanced via {k}x original duplication)"
        )
        run_condition(experiment_id, f"interp_ablation_k{k}", combined_train, dataset_a_dev)


if __name__ == "__main__":
    main()
