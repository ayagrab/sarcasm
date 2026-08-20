"""Phase 9: consolidate the learning-curve points (0/10/25/50/75/100% of
SIGN Train, primary condition) into one table + plot per model. Reads
already-persisted results only -- no new inference. Endpoints (0% and
100%) come from Phase 4's zero-transfer and Phase 8's condition B, reused
not rerun; 10/25/50/75% come from this phase's own runs. Tolerant of
missing fractions (e.g. an interrupted VM run) -- reports what's
available rather than failing.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config.sign_settings import sign_settings
from src.sign.data.load_sign import load_family_table
from src.sign.family_eval.run_family_eval import merge_with_family_table, task_a_block

# (experiment_id, is_dual_target) -- dual-target experiments (Phase 8/9/10)
# store metrics under sign_test/, zero-transfer (Phase 4) stores them flat.
M1_POINTS = {
    0.0: ("EXP-SIGN-011", False),
    0.10: ("EXP-SIGN-025", True),
    0.25: ("EXP-SIGN-026", True),
    0.50: ("EXP-SIGN-027", True),
    0.75: ("EXP-SIGN-028", True),
    1.0: ("EXP-SIGN-021", True),
}
M6_POINTS = {
    0.0: ("EXP-SIGN-016", False),
    0.10: ("EXP-SIGN-029", True),
    0.25: ("EXP-SIGN-030", True),
    0.50: ("EXP-SIGN-031", True),
    0.75: ("EXP-SIGN-032", True),
    1.0: ("EXP-SIGN-023", True),
}


def load_point_metrics(experiment_id: str, is_dual_target: bool, family_table: pd.DataFrame) -> dict | None:
    base = sign_settings.results_dir / experiment_id
    metrics_path = (base / "sign_test" / "metrics.json") if is_dual_target else (base / "metrics.json")
    predictions_path = (base / "sign_test" / "predictions.csv") if is_dual_target else (base / "predictions.csv")
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
    for frac, (experiment_id, is_dual_target) in sorted(points.items()):
        result = load_point_metrics(experiment_id, is_dual_target, family_table)
        if result is None:
            continue
        rows.append({"model": model_name, "sign_train_fraction": frac, "experiment_id": experiment_id, **result})
    return pd.DataFrame(rows)


def plot_curve(summary_df: pd.DataFrame, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model_name, group in summary_df.groupby("model"):
        group = group.sort_values("sign_train_fraction")
        axes[0].plot(group["sign_train_fraction"] * 100, group["task_b_macro_f1"], marker="o", label=model_name)
        axes[1].plot(group["sign_train_fraction"] * 100, group["task_a_detection_rate"], marker="o", label=model_name)
    axes[0].set_title("Task B: SIGN Test Macro F1 vs. SIGN Train fraction")
    axes[0].set_xlabel("SIGN Train used (%)")
    axes[0].set_ylabel("Macro F1")
    axes[1].set_title("Task A: sarcasm detection rate vs. SIGN Train fraction")
    axes[1].set_xlabel("SIGN Train used (%)")
    axes[1].set_ylabel("Detection rate")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    family_table = load_family_table("test")
    m1_summary = build_summary(M1_POINTS, family_table, "M1")
    m6_summary = build_summary(M6_POINTS, family_table, "M6")
    combined = pd.concat([m1_summary, m6_summary], ignore_index=True)

    out_dir = sign_settings.results_dir / "learning_curve"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "summary.csv", index=False)

    m1_missing = sorted(set(M1_POINTS) - set(m1_summary["sign_train_fraction"])) if len(m1_summary) else sorted(M1_POINTS)
    m6_missing = sorted(set(M6_POINTS) - set(m6_summary["sign_train_fraction"])) if len(m6_summary) else sorted(M6_POINTS)
    print(f"M1: {len(m1_summary)}/6 points available. Missing: {m1_missing}")
    print(f"M6: {len(m6_summary)}/6 points available. Missing: {m6_missing}")

    if len(combined):
        plot_curve(combined, out_dir / "learning_curve.png")
        print(f"Saved: {out_dir / 'summary.csv'}, {out_dir / 'learning_curve.png'}")
    else:
        print("No points available yet -- nothing to plot.")


if __name__ == "__main__":
    main()
