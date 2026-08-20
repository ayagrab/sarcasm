"""Phase 2 (SIGN_GENERALIZATION_PLAN.md): dataset characterization.

Compares three corpora, always kept separate (never merges SIGN into one
blob):

    A  -- Dataset A (Part II's ~9,386-sentence classification corpus,
          train+dev+test combined, both labels)
    B1 -- SIGN sarcastic originals (train+dev+test combined)
    B2 -- SIGN non-sarcastic human interpretations (train+dev+test combined)

Writes `results/sign/characterization/corpus_stats.json` (all text-level
stats), `embeddings_2d.csv` (sampled PCA/UMAP coordinates), and PNG
figures under `results/sign/characterization/figures/`. No model
training, no VM.

Run: `python -m src.sign.characterization.run_characterization`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.classification_settings import classification_settings
from config.sign_settings import sign_settings
from src.sign.characterization.embeddings import embed_texts, reduce_dimensions, sample_for_embedding
from src.sign.characterization.stats import describe_corpus
from src.sign.data.load_sign import load_family_table

OUTPUT_DIR = sign_settings.results_dir / "characterization"
FIGURES_DIR = OUTPUT_DIR / "figures"
GROUP_ORDER = ["Dataset A", "SIGN originals", "SIGN interpretations"]
PALETTE = dict(zip(GROUP_ORDER, sns.color_palette("Set2", 3)))


def load_dataset_a() -> pd.DataFrame:
    frames = []
    for split in ("train", "dev", "test"):
        frames.append(pd.read_csv(classification_settings.splits_dir / f"{split}.csv", encoding="utf-8-sig"))
    df = pd.concat(frames, ignore_index=True)
    return df[["example_id", "text", "label"]]


def load_sign_roles() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [load_family_table(split) for split in ("train", "dev", "test")]
    df = pd.concat(frames, ignore_index=True)
    originals = df[df["role"] == "original"][["example_id", "text", "label", "family_id", "split"]]
    interpretations = df[df["role"] == "interpretation"][["example_id", "text", "label", "family_id", "split"]]
    return originals, interpretations


def class_structure(dataset_a: pd.DataFrame, sign_orig: pd.DataFrame, sign_interp: pd.DataFrame) -> dict:
    return {
        "Dataset A": dataset_a["label"].value_counts().to_dict(),
        "SIGN originals": sign_orig["label"].value_counts().to_dict(),
        "SIGN interpretations": sign_interp["label"].value_counts().to_dict(),
        "note": (
            "SIGN originals are ALL sarcastic by construction (n="
            f"{len(sign_orig)}); SIGN interpretations are ALL not_sarcastic "
            f"by construction (n={len(sign_interp)}). SIGN is never a "
            "15,000-example balanced/independent sarcastic set -- see "
            "SIGN_GENERALIZATION_PLAN.md section 1."
        ),
    }


def plot_length_distributions(groups: dict[str, pd.Series]) -> None:
    rows = []
    for name, texts in groups.items():
        for wl in texts.astype(str).str.split().apply(len):
            rows.append({"group": name, "word_length": wl})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="word_length", hue="group", hue_order=GROUP_ORDER, palette=PALETTE, element="step", stat="density", common_norm=False, bins=40)
    plt.xlim(0, df["word_length"].quantile(0.99))
    plt.title("Word-length distribution")
    plt.xlabel("Words per example")
    plt.savefig(FIGURES_DIR / "length_distribution.png", bbox_inches="tight")
    plt.close()


def plot_punctuation_case_comparison(stats_by_group: dict[str, dict]) -> None:
    rows = []
    fields = ["question_mark_rate", "exclamation_mark_rate", "any_punctuation_rate", "any_uppercase_char_rate"]
    for name, stats in stats_by_group.items():
        for field in fields:
            rows.append({"group": name, "feature": field, "rate": stats["punctuation_and_case"][field]})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="feature", y="rate", hue="group", hue_order=GROUP_ORDER, palette=PALETTE)
    plt.title("Punctuation / capitalization presence")
    plt.ylabel("Fraction of examples")
    plt.xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.legend(title="")
    plt.savefig(FIGURES_DIR / "punctuation_case_comparison.png", bbox_inches="tight")
    plt.close()


def plot_sentiment_comparison(stats_by_group: dict[str, dict]) -> None:
    rows = []
    for name, stats in stats_by_group.items():
        rows.append({"group": name, "compound_mean": stats["sentiment"]["compound_mean"], "compound_std": stats["sentiment"]["compound_std"]})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="group", y="compound_mean", order=GROUP_ORDER, palette=PALETTE)
    plt.errorbar(x=range(len(df)), y=df.set_index("group").loc[GROUP_ORDER, "compound_mean"], yerr=df.set_index("group").loc[GROUP_ORDER, "compound_std"], fmt="none", ecolor="black", capsize=4)
    plt.title("Mean VADER sentiment (compound score)")
    plt.ylabel("Compound sentiment (-1 to 1)")
    plt.xlabel("")
    plt.savefig(FIGURES_DIR / "sentiment_comparison.png", bbox_inches="tight")
    plt.close()


def run_embedding_analysis(groups: dict[str, pd.Series], seed: int, n_per_group: int) -> pd.DataFrame:
    df = pd.concat(
        [pd.DataFrame({"text": texts.astype(str).values, "group": name}) for name, texts in groups.items()],
        ignore_index=True,
    )
    sampled = sample_for_embedding(df, text_col="text", group_col="group", n_per_group=n_per_group, seed=seed)

    print(f"  embedding {len(sampled)} sampled texts ({dict(sampled['group'].value_counts())}) ...")
    vectors = embed_texts(sampled["text"].tolist())
    reduced = reduce_dimensions(vectors, seed=seed)

    out = sampled.reset_index(drop=True).copy()
    out["pca_x"], out["pca_y"] = reduced["pca"][:, 0], reduced["pca"][:, 1]
    if reduced["umap"] is not None:
        out["umap_x"], out["umap_y"] = reduced["umap"][:, 0], reduced["umap"][:, 1]

    for proj, xcol, ycol in [("pca", "pca_x", "pca_y")] + ([("umap", "umap_x", "umap_y")] if reduced["umap"] is not None else []):
        plt.figure(figsize=(7, 6))
        sns.scatterplot(data=out, x=xcol, y=ycol, hue="group", hue_order=GROUP_ORDER, palette=PALETTE, alpha=0.5, s=18)
        plt.title(f"Sentence-embedding {proj.upper()} projection")
        plt.legend(title="")
        plt.savefig(FIGURES_DIR / f"embedding_{proj}.png", bbox_inches="tight")
        plt.close()

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: dataset characterization (Dataset A vs. SIGN).")
    parser.add_argument("--seed", type=int, default=sign_settings.random_seed)
    parser.add_argument("--embedding-sample-per-group", type=int, default=2000)
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading corpora ...")
    dataset_a = load_dataset_a()
    sign_orig, sign_interp = load_sign_roles()
    groups_text = {
        "Dataset A": dataset_a["text"],
        "SIGN originals": sign_orig["text"],
        "SIGN interpretations": sign_interp["text"],
    }

    print("Computing text stats ...")
    stats_by_group = {name: describe_corpus(texts, seed=args.seed) for name, texts in groups_text.items()}
    class_struct = class_structure(dataset_a, sign_orig, sign_interp)

    with open(OUTPUT_DIR / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump({"class_structure": class_struct, "corpus_stats": stats_by_group}, f, indent=2, ensure_ascii=False)
    print(f"  -> {OUTPUT_DIR / 'corpus_stats.json'}")

    print("Plotting length / punctuation / sentiment comparisons ...")
    plot_length_distributions(groups_text)
    plot_punctuation_case_comparison(stats_by_group)
    plot_sentiment_comparison(stats_by_group)

    if not args.skip_embeddings:
        print("Computing sentence embeddings + PCA/UMAP ...")
        embed_df = run_embedding_analysis(groups_text, seed=args.seed, n_per_group=args.embedding_sample_per_group)
        embed_df.to_csv(OUTPUT_DIR / "embeddings_2d.csv", index=False, encoding="utf-8-sig")
        print(f"  -> {OUTPUT_DIR / 'embeddings_2d.csv'}")

    print(f"Figures saved under {FIGURES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
