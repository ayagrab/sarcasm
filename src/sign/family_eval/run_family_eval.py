"""Phase 5: SIGN contrastive / family-aware evaluation
(SIGN_GENERALIZATION_PLAN.md, Phase 5). Consumes Phase 4's already-persisted
`results/sign/EXP-SIGN-0{11..16}/predictions.csv` -- no new model inference,
no SIGN Train use. For each method, computes and reports (never conflated,
see the plan's Task A/B/Primary-Reference clarification):

- Task A: SIGN originals only (sarcasm detection rate is primary).
- Task B: full 1,735-row contrastive set (Macro F1 is primary).
- Primary-Reference: original vs. interpretation #1 only, plus pair
  success rate.
- Per-interpretation-rank breakdown: not_sarcastic recall by rank 1-5.
- View 1 (primary-reference family view) / View 2 (full-family view,
  all families and clean-only), via `src.sign.family_eval.metrics`.

Writes one JSON per method under `results/sign/family_eval/<EXP-ID>/`
plus a single consolidated `results/sign/family_eval/m1_m6_comparison.csv`.
"""
from __future__ import annotations

import json

import pandas as pd

from config.classification_settings import classification_settings
from config.sign_settings import sign_settings
from src.classification.evaluation.metrics import compute_metrics
from src.sign.data.load_sign import load_family_table
from src.sign.family_eval.metrics import compute_family_metrics

METHOD_EXPERIMENT_IDS = {
    "M1_tfidf_logreg": "EXP-SIGN-011",
    "M2_qwen_zero_shot": "EXP-SIGN-012",
    "M3_qwen_few_shot": "EXP-SIGN-013",
    "M4_qwen_reasoning": "EXP-SIGN-014",
    "M5_dspy_frozen": "EXP-SIGN-015",
    "M6_deberta": "EXP-SIGN-016",
}

FAMILY_JOIN_COLUMNS = [
    "example_id",
    "family_id",
    "role",
    "interp_index",
    "is_primary_interpretation",
    "is_clean_family",
]
FAMILY_METRIC_COLUMNS = ["family_id", "role", "predicted_label"]


def load_predictions(experiment_id: str) -> pd.DataFrame:
    path = sign_settings.results_dir / experiment_id / "predictions.csv"
    return pd.read_csv(path, encoding="utf-8-sig")


def merge_with_family_table(predictions: pd.DataFrame, family_table: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(family_table[FAMILY_JOIN_COLUMNS], on="example_id", how="left")
    if merged["family_id"].isna().any():
        missing = merged.loc[merged["family_id"].isna(), "example_id"].tolist()
        raise ValueError(f"Predictions rows with no matching family_table entry: {missing[:5]}")
    return merged


def task_a_block(merged: pd.DataFrame) -> dict:
    originals = merged[merged["role"] == "original"]
    n = len(originals)
    n_correct = int((originals["predicted_label"] == "sarcastic").sum())
    rate = n_correct / n if n else float("nan")
    return {
        "n_originals": n,
        "n_correctly_predicted_sarcastic": n_correct,
        "sarcasm_detection_rate": rate,
        "false_negative_rate": (1.0 - rate) if n else float("nan"),
    }


def task_b_block(predictions: pd.DataFrame) -> dict:
    return compute_metrics(
        predictions,
        labels=classification_settings.labels,
        positive_label=classification_settings.positive_label,
    )


def primary_reference_block(merged: pd.DataFrame) -> dict:
    subset = merged[(merged["role"] == "original") | (merged["is_primary_interpretation"])]
    metrics = compute_metrics(
        subset,
        labels=classification_settings.labels,
        positive_label=classification_settings.positive_label,
    )
    originals = subset[subset["role"] == "original"][["family_id", "predicted_label"]].rename(
        columns={"predicted_label": "orig_pred"}
    )
    interp1 = subset[subset["role"] == "interpretation"][["family_id", "predicted_label"]].rename(
        columns={"predicted_label": "interp1_pred"}
    )
    pairs = originals.merge(interp1, on="family_id", how="inner")
    if len(pairs):
        pair_success = float(
            ((pairs["orig_pred"] == "sarcastic") & (pairs["interp1_pred"] == "not_sarcastic")).mean()
        )
    else:
        pair_success = float("nan")
    metrics["pair_success_rate"] = pair_success
    metrics["n_pairs"] = int(len(pairs))
    return metrics


def per_interpretation_rank_recall(merged: pd.DataFrame) -> dict:
    interps = merged[merged["role"] == "interpretation"]
    out = {}
    for rank in range(1, 6):
        rows = interps[interps["interp_index"] == rank]
        if len(rows) == 0:
            out[f"rank_{rank}"] = {"n": 0, "not_sarcastic_recall": float("nan")}
            continue
        recall = float((rows["predicted_label"] == "not_sarcastic").mean())
        out[f"rank_{rank}"] = {"n": int(len(rows)), "not_sarcastic_recall": recall}
    return out


def family_views(merged: pd.DataFrame) -> dict:
    view1_df = merged[(merged["role"] == "original") | (merged["is_primary_interpretation"])]
    view2_all_df = merged
    view2_clean_df = merged[merged["is_clean_family"]]
    return {
        "view1_primary_reference_family": compute_family_metrics(view1_df[FAMILY_METRIC_COLUMNS]),
        "view2_full_family_all": compute_family_metrics(view2_all_df[FAMILY_METRIC_COLUMNS]),
        "view2_full_family_clean_only": compute_family_metrics(view2_clean_df[FAMILY_METRIC_COLUMNS]),
    }


def run_method(method_name: str, experiment_id: str, family_table: pd.DataFrame) -> dict:
    predictions = load_predictions(experiment_id)
    merged = merge_with_family_table(predictions, family_table)
    result = {
        "method": method_name,
        "experiment_id": experiment_id,
        "task_a": task_a_block(merged),
        "task_b": task_b_block(predictions),
        "primary_reference": primary_reference_block(merged),
        "per_interpretation_rank": per_interpretation_rank_recall(merged),
        **family_views(merged),
    }
    out_dir = sign_settings.results_dir / "family_eval" / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(
        f"[{experiment_id}] {method_name}: "
        f"Task A detection={result['task_a']['sarcasm_detection_rate']:.4f} "
        f"Task B macro_f1={result['task_b']['macro_f1']:.4f} "
        f"primary_ref pair_success={result['primary_reference']['pair_success_rate']:.4f}"
    )
    return result


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "method": r["method"],
                "experiment_id": r["experiment_id"],
                "task_a_detection_rate": r["task_a"]["sarcasm_detection_rate"],
                "task_a_fn_rate": r["task_a"]["false_negative_rate"],
                "task_b_macro_f1": r["task_b"]["macro_f1"],
                "task_b_accuracy": r["task_b"]["accuracy"],
                "primary_ref_macro_f1": r["primary_reference"]["macro_f1"],
                "primary_ref_accuracy": r["primary_reference"]["accuracy"],
                "primary_ref_pair_success_rate": r["primary_reference"]["pair_success_rate"],
                "rank1_not_sarcastic_recall": r["per_interpretation_rank"]["rank_1"]["not_sarcastic_recall"],
                "rank2_not_sarcastic_recall": r["per_interpretation_rank"]["rank_2"]["not_sarcastic_recall"],
                "rank3_not_sarcastic_recall": r["per_interpretation_rank"]["rank_3"]["not_sarcastic_recall"],
                "rank4_not_sarcastic_recall": r["per_interpretation_rank"]["rank_4"]["not_sarcastic_recall"],
                "rank5_not_sarcastic_recall": r["per_interpretation_rank"]["rank_5"]["not_sarcastic_recall"],
                "view1_pairwise_contrastive_accuracy": r["view1_primary_reference_family"][
                    "pairwise_contrastive_accuracy"
                ],
                "view1_strict_family_accuracy": r["view1_primary_reference_family"]["strict_family_accuracy"],
                "view2_all_pairwise_contrastive_accuracy": r["view2_full_family_all"][
                    "pairwise_contrastive_accuracy"
                ],
                "view2_all_strict_family_accuracy": r["view2_full_family_all"]["strict_family_accuracy"],
                "view2_all_soft_family_score": r["view2_full_family_all"]["soft_family_score"]["mean_score"],
                "view2_clean_strict_family_accuracy": r["view2_full_family_clean_only"]["strict_family_accuracy"],
                "view2_clean_soft_family_score": r["view2_full_family_clean_only"]["soft_family_score"][
                    "mean_score"
                ],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    family_table = load_family_table("test")
    results = [
        run_method(method_name, experiment_id, family_table)
        for method_name, experiment_id in METHOD_EXPERIMENT_IDS.items()
    ]
    comparison = build_comparison_table(results)
    out_dir = sign_settings.results_dir / "family_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / "m1_m6_comparison.csv", index=False)
    print(f"Saved: {out_dir / 'm1_m6_comparison.csv'}")


if __name__ == "__main__":
    main()
