"""Phase 10: consolidate the interpretation-count ablation (k=1/2/3/5,
same 100% SIGN Train family set, primary condition reused for k=1) into
one table per model. Reads already-persisted results only. Tolerant of
missing k's (e.g. VM not yet run for M6).
"""
from __future__ import annotations

import json

import pandas as pd

from config.sign_settings import sign_settings
from src.sign.data.load_sign import load_family_table
from src.sign.family_eval.run_family_eval import merge_with_family_table, task_a_block

# k=1 reuses Phase 8's condition B (dual-target); k=2/3/5 are this
# phase's own runs (also dual-target).
M1_POINTS = {1: "EXP-SIGN-021", 2: "EXP-SIGN-033", 3: "EXP-SIGN-034", 5: "EXP-SIGN-035"}
M6_POINTS = {1: "EXP-SIGN-023", 2: "EXP-SIGN-036", 3: "EXP-SIGN-037", 5: "EXP-SIGN-038"}


def load_point_metrics(experiment_id: str, family_table: pd.DataFrame) -> dict | None:
    base = sign_settings.results_dir / experiment_id
    metrics_path = base / "sign_test" / "metrics.json"
    predictions_path = base / "sign_test" / "predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        return None
    metrics = json.load(open(metrics_path, encoding="utf-8"))
    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig")
    merged = merge_with_family_table(predictions, family_table)
    task_a = task_a_block(merged)
    return {
        "task_b_macro_f1": metrics["macro_f1"],
        "task_b_accuracy": metrics["accuracy"],
        "task_a_detection_rate": task_a["sarcasm_detection_rate"],
    }


def build_summary(points: dict, family_table: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for k, experiment_id in sorted(points.items()):
        result = load_point_metrics(experiment_id, family_table)
        if result is None:
            continue
        rows.append({"model": model_name, "k_interpretations": k, "experiment_id": experiment_id, **result})
    return pd.DataFrame(rows)


def main() -> None:
    family_table = load_family_table("test")
    m1_summary = build_summary(M1_POINTS, family_table, "M1")
    m6_summary = build_summary(M6_POINTS, family_table, "M6")
    combined = pd.concat([m1_summary, m6_summary], ignore_index=True)

    out_dir = sign_settings.results_dir / "interp_count_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "summary.csv", index=False)

    m1_missing = sorted(set(M1_POINTS) - set(m1_summary["k_interpretations"])) if len(m1_summary) else sorted(M1_POINTS)
    m6_missing = sorted(set(M6_POINTS) - set(m6_summary["k_interpretations"])) if len(m6_summary) else sorted(M6_POINTS)
    print(f"M1: {len(m1_summary)}/4 points available. Missing k={m1_missing}")
    print(f"M6: {len(m6_summary)}/4 points available. Missing k={m6_missing}")
    print(f"Saved: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
