"""Phase 9, M6 leg: learning curve -- how much SIGN Train is needed
(SIGN_GENERALIZATION_PLAN.md, Phase 9). RQ3. Same design as the M1 leg:
sweeps 10/25/50/75% of SIGN Train families (primary balanced condition
only), family-level sampling. 0% and 100% are Phase 4's zero-transfer
(EXP-SIGN-016) and Phase 8's condition B (EXP-SIGN-023), reused not
rerun. Same recipe as condition B throughout: Dataset A TRAIN (full) +
the SIGN Train fraction, combined fit, early-stopped on Dataset A Dev,
evaluated on SIGN Test and Dataset A TEST.

Reuses `run_m6_domain_adaptation.run_condition` directly -- it already
takes arbitrary train_df/dev_df, no duplication needed.

VM required (GPU fine-tuning). Run:
`python -m src.sign.learning_curve.run_m6_learning_curve`
"""
from __future__ import annotations

import pandas as pd

from config.classification_settings import classification_settings
from src.sign.data.load_sign import load_family_table
from src.sign.domain_adaptation.run_m6_domain_adaptation import run_condition
from src.sign.learning_curve.run_m1_learning_curve import build_sign_fraction

FRACTIONS_TO_RUN = {
    0.10: "EXP-SIGN-029",
    0.25: "EXP-SIGN-030",
    0.50: "EXP-SIGN-031",
    0.75: "EXP-SIGN-032",
}


def main() -> None:
    seed = classification_settings.random_seed
    dataset_a_train = pd.read_csv(classification_settings.splits_dir / "train.csv", encoding="utf-8-sig")
    dataset_a_dev = pd.read_csv(classification_settings.splits_dir / "dev.csv", encoding="utf-8-sig")
    train_table = load_family_table("train")

    for frac, experiment_id in FRACTIONS_TO_RUN.items():
        sign_fraction = build_sign_fraction(train_table, frac, seed)
        combined_train = pd.concat(
            [dataset_a_train[["example_id", "text", "label"]], sign_fraction[["example_id", "text", "label"]]],
            ignore_index=True,
        )
        print(f"frac={frac}: {experiment_id} -- {len(combined_train)} train rows ({len(sign_fraction)} from SIGN)")
        run_condition(experiment_id, f"learning_curve_frac_{frac}", combined_train, dataset_a_dev)


if __name__ == "__main__":
    main()
