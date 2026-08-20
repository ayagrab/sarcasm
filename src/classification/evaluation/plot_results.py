"""Figures for Part II (sarcasm detection) -- the six-method comparison.

Part I (interpretation) already has figures under data/summaries/figures/,
produced by src/postprocessing/*.py. Part II never got equivalent plots --
its results only ever became tables (results/EXP-*/metrics.json,
results/cross_model_test_analysis.csv). This script closes that gap by
building the standard figures directly from those already-computed
artifacts -- no new experiments, no new numbers, only visualizing what
PROJECT_SUMMARY.md / EXPERIMENT_LOG.md / the report already report as text.

Style matches src/postprocessing/plot_text_metrics.py and
human_llm_agreement.py: matplotlib + seaborn, Agg backend, "Set2" for
categorical comparisons, "Blues" (single sequential hue) for heatmaps.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.classification_settings import classification_settings
from src.classification.evaluation.io import load_experiment_metrics

# (method_id, display_label, TEST experiment_id, DEV experiment_id, cross-model CSV column suffix)
METHODS = [
    ("M1", "M1 TF-IDF + LR", "EXP-001", "EXP-001-dev-ref", "tfidf"),
    ("M2", "M2 Qwen zero-shot", "EXP-002-TEST", "EXP-002", "zero_shot"),
    ("M3", "M3 Qwen few-shot", "EXP-003-TEST", "EXP-003", "fewshot_random"),
    ("M4", "M4 Qwen reasoning", "EXP-005-TEST", "EXP-005", "reasoning"),
    ("M5", "M5 DSPy MIPROv2", "EXP-008-TEST", "EXP-008", "dspy_mipro"),
    ("M6", "M6 DeBERTa-v3-base", "EXP-009-TEST", "EXP-009", "deberta"),
]
ORDER = ["M6 DeBERTa-v3-base", "M1 TF-IDF + LR", "M5 DSPy MIPROv2", "M2 Qwen zero-shot", "M3 Qwen few-shot", "M4 Qwen reasoning"]


def plot_macro_f1_comparison(output_dir: Path, results_dir: Path) -> None:
    rows = []
    for _, label, test_id, dev_id, _ in METHODS:
        rows.append({"method": label, "split": "TEST", "macro_f1": load_experiment_metrics(test_id, results_dir)["macro_f1"]})
        rows.append({"method": label, "split": "DEV", "macro_f1": load_experiment_metrics(dev_id, results_dir)["macro_f1"]})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="method", y="macro_f1", hue="split", order=ORDER, palette="Set2", hue_order=["DEV", "TEST"])
    plt.title("Macro F1 by Method: DEV vs. Sealed TEST", fontsize=14)
    plt.ylabel("Macro F1")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.legend(title="")
    plt.savefig(output_dir / "macro_f1_comparison.png", bbox_inches="tight")
    plt.close()


def plot_per_class_f1(output_dir: Path, results_dir: Path) -> None:
    rows = []
    for _, label, test_id, _, _ in METHODS:
        m = load_experiment_metrics(test_id, results_dir)
        for cls, f1 in [("sarcastic", m["per_class"]["sarcastic"]["f1"]), ("not_sarcastic", m["per_class"]["not_sarcastic"]["f1"])]:
            rows.append({"method": label, "class": cls, "f1": f1})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="method", y="f1", hue="class", order=ORDER, palette="Set2", hue_order=["sarcastic", "not_sarcastic"])
    plt.title("Per-Class F1 on TEST -- the Sarcastic-Over-Prediction Bias", fontsize=14)
    plt.ylabel("F1")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.legend(title="")
    plt.savefig(output_dir / "per_class_f1.png", bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(output_dir: Path, results_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (method_id, label, test_id, _, _) in zip(axes, [m for m in METHODS if m[0] in ("M1", "M6")]):
        cm_data = load_experiment_metrics(test_id, results_dir)["confusion_matrix"]
        labels = cm_data["labels"]
        sns.heatmap(cm_data["matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False, ax=ax, annot_kws={"size": 13})
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold" if method_id == "M1" else "")
    fig.suptitle("Confusion Matrices on TEST: the Two Label-Trained Methods", fontsize=14, y=1.03)
    fig.savefig(output_dir / "confusion_matrices_m1_m6.png", bbox_inches="tight")
    plt.close(fig)


def plot_category_breakdown(output_dir: Path, cross_model_path: Path) -> None:
    df = pd.read_csv(cross_model_path)
    rows = []
    for method_id, label, _, _, suffix in METHODS:
        col = f"correct_{method_id}_{suffix}"
        per_cat = df.groupby("category")[col].mean()
        for cat, acc in per_cat.items():
            rows.append({"method": label, "category": cat, "accuracy": acc})
    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(11, 6))
    sns.barplot(data=plot_df, x="category", y="accuracy", hue="method", hue_order=ORDER, palette="Set2", order=["GEN", "HYP", "RQ"])
    plt.title("Accuracy by Sarcasm Category (TEST)", fontsize=14)
    plt.ylabel("Accuracy")
    plt.xlabel("")
    plt.ylim(0, 1)
    plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.savefig(output_dir / "category_breakdown.png", bbox_inches="tight")
    plt.close()


def plot_cross_model_agreement(output_dir: Path, cross_model_path: Path) -> None:
    df = pd.read_csv(cross_model_path)
    pred_cols = {label: f"M{method_id[1]}_{suffix}" for method_id, label, _, _, suffix in METHODS}
    n = len(METHODS)
    agreement = pd.DataFrame(index=list(pred_cols), columns=list(pred_cols), dtype=float)
    for label_a, col_a in pred_cols.items():
        for label_b, col_b in pred_cols.items():
            agreement.loc[label_a, label_b] = (df[col_a] == df[col_b]).mean()

    plt.figure(figsize=(8, 6.5))
    short_labels = [m[1].split(" ", 1)[1] for m in METHODS]
    sns.heatmap(
        agreement.values, annot=True, fmt=".0%", cmap="Blues", vmin=0.5, vmax=1.0,
        xticklabels=short_labels, yticklabels=short_labels, cbar_kws={"label": "Pairwise prediction agreement"},
    )
    plt.title("Cross-Model Prediction Agreement on TEST", fontsize=14)
    plt.xticks(rotation=30, ha="right")
    plt.savefig(output_dir / "cross_model_agreement.png", bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=classification_settings.results_dir)
    parser.add_argument("--cross-model-csv", type=Path, default=classification_settings.results_dir / "cross_model_test_analysis.csv")
    parser.add_argument("--output-dir", type=Path, default=classification_settings.results_dir / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("ggplot")

    plot_macro_f1_comparison(args.output_dir, args.results_dir)
    plot_per_class_f1(args.output_dir, args.results_dir)
    plot_confusion_matrices(args.output_dir, args.results_dir)
    plot_category_breakdown(args.output_dir, args.cross_model_csv)
    plot_cross_model_agreement(args.output_dir, args.cross_model_csv)
    print(f"Saved 5 plots to {args.output_dir}")


if __name__ == "__main__":
    main()
