"""Compare the 3 human annotators against each other and against the LLM judge.

Builds its data by joining data/alt_test/{humans,llm}_annotations.json with
the model interpretation text in data/model_outputs/ -- no separate copy of
this data is stored anywhere.

Reproduces the findings in docs/project_history.md (2026-07-16
meeting): Fleiss' Kappa ~0.282 among human annotators (fair agreement,
confirming sarcasm scoring is subjective even for humans); Nvidia leads
human-rated quality with Liquid a distant last; ChatGPT is a harsher judge
than humans on average and rarely uses the middle (2) score.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from config.settings import settings
from src.common.file_utils import ensure_parent_dir


def _parse_instance_key(key: str) -> tuple[str, str, str]:
    """Split an instance id like '1_gemini_<sentence>' into (folder, model, sentence)."""
    folder, model, sentence = key.split("_", 2)
    return folder, model, sentence


def load_human_scores(humans_path: Path) -> pd.DataFrame:
    """Long format: one row per (human annotator, instance)."""
    humans_data = json.loads(humans_path.read_text(encoding="utf-8"))
    rows = [
        {"human": human, "folder": folder, "model": model, "sentence": sentence, "score": score}
        for human, annotations in humans_data.items()
        for key, score in annotations.items()
        for folder, model, sentence in [_parse_instance_key(key)]
    ]
    return pd.DataFrame(rows)


def load_llm_scores(llm_path: Path) -> pd.DataFrame:
    """One row per instance: the LLM judge's score."""
    llm_data = json.loads(llm_path.read_text(encoding="utf-8"))
    rows = [
        {"folder": folder, "model": model, "sentence": sentence, "score": score}
        for key, score in llm_data.items()
        for folder, model, sentence in [_parse_instance_key(key)]
    ]
    return pd.DataFrame(rows)


def attach_interpretations(df: pd.DataFrame, model_outputs_dir: Path) -> pd.DataFrame:
    """Look up model_interpretation text from data/model_outputs/ for each (folder, model, sentence) row."""
    lookups = {}
    for folder in df["folder"].unique():
        for model in df["model"].unique():
            path = model_outputs_dir / f"experiment_{int(folder):02d}" / f"{model}_run_{int(folder):02d}.csv"
            if path.is_file():
                out_df = pd.read_csv(path, encoding="utf-8-sig")
                lookups[(folder, model)] = dict(zip(out_df["sarcastic_sentence"], out_df["model_interpretation"]))

    def lookup(row) -> str | None:
        return lookups.get((row["folder"], row["model"]), {}).get(row["sentence"])

    df = df.copy()
    df["model_interpretation"] = df.apply(lookup, axis=1)
    return df


def compute_fleiss_kappa(human_df: pd.DataFrame) -> float:
    """Inter-rater agreement among the human annotators."""
    pivot = human_df.pivot_table(index=["folder", "model", "sentence"], columns="human", values="score").dropna()
    data_counts, _ = aggregate_raters(pivot.values)
    return fleiss_kappa(data_counts)


def build_comparison_table(humans_path: Path, llm_path: Path) -> pd.DataFrame:
    """One row per instance with the average human score and the LLM (judge) score."""
    human_df = load_human_scores(humans_path)
    llm_df = load_llm_scores(llm_path)

    human_avg = human_df.groupby(["folder", "model", "sentence"])["score"].mean().reset_index()
    human_avg = human_avg.rename(columns={"score": "human_score"})

    merged = pd.merge(llm_df.rename(columns={"score": "chatgpt_score"}), human_avg, on=["folder", "model", "sentence"], how="inner")
    merged["human_score_rounded"] = merged["human_score"].round().astype(int)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare human annotators against each other and against the LLM judge.")
    parser.add_argument("--alt-test-dir", type=Path, default=settings.data_dir / "alt_test")
    parser.add_argument("--outputs-dir", type=Path, default=settings.model_outputs_dir)
    parser.add_argument("--figures-dir", type=Path, default=settings.summaries_dir / "figures")
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("ggplot")

    human_df = load_human_scores(args.alt_test_dir / "humans_annotations.json")
    kappa = compute_fleiss_kappa(human_df)
    print(f"Fleiss' Kappa (human annotators): {kappa:.3f}")
    print("0.21-0.40 = fair agreement -- scoring sarcasm translations is subjective even for humans.\n")

    merged = build_comparison_table(args.alt_test_dir / "humans_annotations.json", args.alt_test_dir / "llm_annotations.json")

    # Plot 1: average score per model, human vs. ChatGPT
    plt.figure(figsize=(10, 6))
    model_scores = merged.groupby("model")[["human_score", "chatgpt_score"]].mean().reset_index()
    model_scores_melted = model_scores.melt(id_vars=["model"], var_name="evaluator", value_name="average_score")
    sns.barplot(data=model_scores_melted, x="model", y="average_score", hue="evaluator", palette="viridis")
    plt.title("Average Translation Score per Model (Human vs. ChatGPT)", fontsize=14)
    plt.ylim(1, 3)
    plt.savefig(args.figures_dir / "1_models_comparison.png", bbox_inches="tight")
    plt.close()

    # Plot 2: average score per prompt, human vs. ChatGPT
    plt.figure(figsize=(10, 6))
    prompt_scores = merged.groupby("folder")[["human_score", "chatgpt_score"]].mean().reset_index()
    prompt_scores_melted = prompt_scores.melt(id_vars=["folder"], var_name="evaluator", value_name="average_score")
    sns.barplot(data=prompt_scores_melted, x="folder", y="average_score", hue="evaluator", palette="mako")
    plt.title("Average Translation Score per Prompt (Human vs. ChatGPT)", fontsize=14)
    plt.ylim(1, 3)
    plt.savefig(args.figures_dir / "2_prompts_comparison.png", bbox_inches="tight")
    plt.close()

    # Plot 3: scatter of ChatGPT vs. human score, with correlation
    plt.figure(figsize=(8, 6))
    correlation = merged["chatgpt_score"].corr(merged["human_score"])
    sns.regplot(
        data=merged, x="chatgpt_score", y="human_score", x_jitter=0.15, y_jitter=0.05,
        scatter_kws={"alpha": 0.6, "color": "teal", "edgecolor": "black", "s": 60},
        line_kws={"color": "darkred", "label": f"Trend line (r={correlation:.2f})"},
    )
    plt.title("ChatGPT Score vs. Human Average Score", fontsize=14)
    plt.legend()
    plt.savefig(args.figures_dir / "3_scatter_plot.png", bbox_inches="tight")
    plt.close()

    # Plot 4: confusion matrix, rounded human score vs. ChatGPT score
    plt.figure(figsize=(7, 5))
    cm = confusion_matrix(merged["human_score_rounded"], merged["chatgpt_score"], labels=[1, 2, 3])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2, 3], yticklabels=[1, 2, 3], cbar=False, annot_kws={"size": 14})
    plt.title("Confusion Matrix: Rounded Human Score vs. ChatGPT Score", fontsize=14)
    plt.xlabel("ChatGPT Predicted Score")
    plt.ylabel("Actual Human Score (Rounded)")
    plt.savefig(args.figures_dir / "4_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    print("--- Agreement rate by model (rounded human score == ChatGPT score) ---")
    agreement = merged["human_score_rounded"] == merged["chatgpt_score"]
    for model, rate in (agreement.groupby(merged["model"]).mean() * 100).round(1).items():
        print(f"  {model}: {rate}%")

    print(f"\nSaved 4 comparison plots to {args.figures_dir}")


if __name__ == "__main__":
    main()
