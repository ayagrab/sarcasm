"""Phase 8's mandatory post-adaptation error-analysis repeat for M6: an
explicit before (Phase 4 zero-transfer, EXP-SIGN-016) vs. after (Phase 8
condition B, EXP-SIGN-023 -- the winning M6 adapted model, Dataset A +
SIGN Train primary) diff. Condition B was chosen over C for the same
reason as M1: it matches C on SIGN Test Macro F1 while completely
avoiding C's catastrophic forgetting of Dataset A (see
PROJECT_SUMMARY.md Phase 8 results).

Run: `python -m src.sign.domain_adaptation.run_m6_post_adaptation_diff`
"""
from __future__ import annotations

import json

import pandas as pd

from config.sign_settings import sign_settings
from src.sign.data.load_sign import load_family_table
from src.sign.domain_adaptation.error_diff import (
    diff_interpretations,
    diff_originals,
    interpretations_correctness,
    originals_correctness,
)

BEFORE_EXPERIMENT_ID = "EXP-SIGN-016"  # Phase 4 zero-transfer
AFTER_EXPERIMENT_ID = "EXP-SIGN-023"  # Phase 8 condition B (winning M6 adapted model)


def main() -> None:
    family_table = load_family_table("test")

    before_predictions = pd.read_csv(
        sign_settings.results_dir / BEFORE_EXPERIMENT_ID / "predictions.csv", encoding="utf-8-sig"
    )
    after_predictions = pd.read_csv(
        sign_settings.results_dir / AFTER_EXPERIMENT_ID / "sign_test" / "predictions.csv", encoding="utf-8-sig"
    )

    before_originals = originals_correctness(before_predictions, family_table)
    after_originals = originals_correctness(after_predictions, family_table)
    originals_diff = diff_originals(before_originals, after_originals)

    before_interps = interpretations_correctness(before_predictions, family_table)
    after_interps = interpretations_correctness(after_predictions, family_table)
    interps_diff = diff_interpretations(before_interps, after_interps)

    result = {
        "model": "M6_deberta",
        "before_experiment_id": BEFORE_EXPERIMENT_ID,
        "after_experiment_id": AFTER_EXPERIMENT_ID,
        "originals_false_negative_diff": originals_diff,
        "interpretations_false_positive_diff": interps_diff,
    }

    out_dir = sign_settings.results_dir / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "post_adaptation_diff_M6.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"Originals (false negatives): {originals_diff['n_fixed']} fixed, "
        f"{originals_diff['n_still_wrong']} still missed, {originals_diff['n_newly_broken']} newly broken "
        f"(of {originals_diff['n_total']})"
    )
    print(
        f"Interpretations (false positives): {interps_diff['n_fixed']} fixed, "
        f"{interps_diff['n_still_wrong']} still wrong, {interps_diff['n_newly_broken']} newly broken "
        f"(of {interps_diff['n_total']})"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
