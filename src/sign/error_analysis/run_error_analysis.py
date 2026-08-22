"""Phase 6: mandatory COMPLETE (not sampled) SIGN error analysis
(PROJECT_SUMMARY.md, Phase 6). Consumes Phase 4's persisted
predictions + Phase 1's family table -- no new model inference, no SIGN
Train use. Builds, for every SIGN original/interpretation with at least
one wrong prediction across all 6 methods (never a curated sample):

- `false_negatives.csv` -- every SIGN original missed by >=1 method, with
  per-model predictions, which/how-many models missed it, all available
  interpretations for context, and per-example text features.
- `false_positives.csv` -- every SIGN interpretation flagged sarcastic by
  >=1 method, mirrored the same way.
- `quantitative_contrast.json` -- missed-vs-always-detected feature
  comparison (length, sentiment, punctuation/case), so a characteristic is
  only reported as explanatory if it's actually more common among misses.
- `cross_model_overlap.json` -- miss-count histogram + pairwise overlap
  between every pair of methods' miss sets.

Qualitative tags in the CSVs are rule-based candidates derived from the
same measurable text features (documented, not manual annotation) -- the
qualitative write-up in the plan/log is a separate step, from actually
reading the hardest (missed-by-most) cases.
"""
from __future__ import annotations

import json

import pandas as pd

from config.sign_settings import sign_settings
from src.sign.characterization.nltk_setup import ensure_nltk_resources
from src.sign.data.load_sign import load_family_table
from src.sign.family_eval.run_family_eval import METHOD_EXPERIMENT_IDS, load_predictions

METHOD_SHORT_NAMES = {
    "M1_tfidf_logreg": "M1",
    "M2_qwen_zero_shot": "M2",
    "M3_qwen_few_shot": "M3",
    "M4_qwen_reasoning": "M4",
    "M5_dspy_frozen": "M5",
    "M6_deberta": "M6",
}
METHOD_COLUMNS = [f"pred_{m}" for m in METHOD_SHORT_NAMES.values()]


def _sentiment_analyzer():
    ensure_nltk_resources()
    from nltk.sentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def text_features(text: str, sia) -> dict:
    t = str(text)
    words = t.split()
    return {
        "word_length": len(words),
        "char_length": len(t),
        "has_question_mark": bool("?" in t),
        "has_exclamation_mark": bool("!" in t),
        "has_all_caps_word": bool(any(w.isupper() and len(w) > 1 for w in words)),
        "uppercase_char_fraction": (sum(1 for c in t if c.isupper()) / len(t)) if len(t) else 0.0,
        "vader_compound": float(sia.polarity_scores(t)["compound"]),
    }


def qualitative_tags(features: dict) -> list[str]:
    """Rule-based candidate tags from measurable text properties (§Phase 6
    hypothesis list) -- a starting scaffold for the qualitative write-up,
    not a substitute for it. Multiple tags may apply; none may apply."""
    tags = []
    if features["has_question_mark"]:
        tags.append("rhetorical_question_candidate")
    if features["has_exclamation_mark"] or features["has_all_caps_word"]:
        tags.append("emphatic_delivery_candidate")
    if not features["has_exclamation_mark"] and not features["has_all_caps_word"] and not features["has_question_mark"]:
        tags.append("flat_delivery_candidate")
    if features["vader_compound"] > 0.3:
        tags.append("positive_surface_wording_candidate")
    if features["word_length"] <= 6:
        tags.append("short_ambiguous_candidate")
    if features["word_length"] >= 25:
        tags.append("long_context_dependent_candidate")
    return tags


def build_wide_predictions(family_table: pd.DataFrame) -> pd.DataFrame:
    """One row per SIGN Test example_id, with a `pred_<M#>` column per
    method, joined to family/role/interp_index/text metadata."""
    wide = family_table[["example_id", "family_id", "role", "interp_index", "text", "label"]].copy()
    for method_name, experiment_id in METHOD_EXPERIMENT_IDS.items():
        short = METHOD_SHORT_NAMES[method_name]
        preds = load_predictions(experiment_id)[["example_id", "predicted_label"]].rename(
            columns={"predicted_label": f"pred_{short}"}
        )
        wide = wide.merge(preds, on="example_id", how="left")
    return wide


def _attach_features_and_tags(df: pd.DataFrame, text_col: str, sia) -> pd.DataFrame:
    feats = df[text_col].apply(lambda t: text_features(t, sia))
    feat_df = pd.DataFrame(list(feats), index=df.index)
    feat_df["qualitative_tags"] = feats.apply(lambda f: ";".join(qualitative_tags(f)))
    return pd.concat([df, feat_df], axis=1)


def build_false_negatives(wide: pd.DataFrame, sia) -> pd.DataFrame:
    originals = wide[wide["role"] == "original"].copy()
    missed_mask = originals[METHOD_COLUMNS].eq("not_sarcastic")
    originals["n_models_missed"] = missed_mask.sum(axis=1)
    originals["which_models_missed"] = missed_mask.apply(
        lambda row: ";".join(m for m, v in zip(METHOD_SHORT_NAMES.values(), row) if v), axis=1
    )
    fn = originals[originals["n_models_missed"] > 0].copy()
    fn = fn.rename(columns={"text": "original_text"})

    interps = wide[wide["role"] == "interpretation"]
    for rank in range(1, 6):
        rank_texts = interps[interps["interp_index"] == rank][["family_id", "text"]].rename(
            columns={"text": f"interpretation_{rank}"}
        )
        fn = fn.merge(rank_texts, on="family_id", how="left")

    fn = _attach_features_and_tags(fn, "original_text", sia)
    fn["notes"] = ""

    ordered_cols = (
        ["family_id", "original_text", "n_models_missed", "which_models_missed"]
        + METHOD_COLUMNS
        + [f"interpretation_{i}" for i in range(1, 6)]
        + ["word_length", "char_length", "has_question_mark", "has_exclamation_mark",
           "has_all_caps_word", "uppercase_char_fraction", "vader_compound", "qualitative_tags", "notes"]
    )
    return fn[ordered_cols].sort_values("n_models_missed", ascending=False).reset_index(drop=True)


def build_false_positives(wide: pd.DataFrame, sia) -> pd.DataFrame:
    interps = wide[wide["role"] == "interpretation"].copy()
    flagged_mask = interps[METHOD_COLUMNS].eq("sarcastic")
    interps["n_models_flagged_sarcastic"] = flagged_mask.sum(axis=1)
    interps["which_models_flagged"] = flagged_mask.apply(
        lambda row: ";".join(m for m, v in zip(METHOD_SHORT_NAMES.values(), row) if v), axis=1
    )
    fp = interps[interps["n_models_flagged_sarcastic"] > 0].copy()
    fp = fp.rename(columns={"text": "interpretation_text", "interp_index": "interp_rank"})

    originals = wide[wide["role"] == "original"][["family_id", "text"]].rename(columns={"text": "original_text"})
    fp = fp.merge(originals, on="family_id", how="left")

    fp = _attach_features_and_tags(fp, "interpretation_text", sia)
    fp["notes"] = ""

    ordered_cols = (
        ["family_id", "interp_rank", "interpretation_text", "original_text",
         "n_models_flagged_sarcastic", "which_models_flagged"]
        + METHOD_COLUMNS
        + ["word_length", "char_length", "has_question_mark", "has_exclamation_mark",
           "has_all_caps_word", "uppercase_char_fraction", "vader_compound", "qualitative_tags", "notes"]
    )
    return fp[ordered_cols].sort_values("n_models_flagged_sarcastic", ascending=False).reset_index(drop=True)


_FEATURE_COLS = [
    "word_length", "char_length", "has_question_mark", "has_exclamation_mark",
    "has_all_caps_word", "uppercase_char_fraction", "vader_compound",
]


def quantitative_contrast(false_negatives: pd.DataFrame, wide: pd.DataFrame, sia) -> dict:
    """Compares text features between originals ever missed (>=1 method)
    and originals always correctly detected (0 methods missed) -- a
    feature is only meaningful if the two groups actually differ."""
    originals = wide[wide["role"] == "original"].copy()
    missed_mask = originals[METHOD_COLUMNS].eq("not_sarcastic")
    originals["n_models_missed"] = missed_mask.sum(axis=1)
    originals = _attach_features_and_tags(originals.rename(columns={"text": "original_text"}), "original_text", sia)

    ever_missed = originals[originals["n_models_missed"] > 0]
    always_detected = originals[originals["n_models_missed"] == 0]

    out = {
        "n_ever_missed": int(len(ever_missed)),
        "n_always_detected": int(len(always_detected)),
        "features": {},
    }
    for col in _FEATURE_COLS:
        out["features"][col] = {
            "ever_missed_mean": float(ever_missed[col].mean()) if len(ever_missed) else float("nan"),
            "always_detected_mean": float(always_detected[col].mean()) if len(always_detected) else float("nan"),
        }
    return out


def cross_model_overlap(wide: pd.DataFrame) -> dict:
    originals = wide[wide["role"] == "original"].copy()
    missed_mask = originals[METHOD_COLUMNS].eq("not_sarcastic")
    n_missed = missed_mask.sum(axis=1)

    histogram = {int(k): int(v) for k, v in n_missed.value_counts().sort_index().items()}

    miss_sets = {
        m: set(originals.loc[missed_mask[f"pred_{m}"], "family_id"]) for m in METHOD_SHORT_NAMES.values()
    }
    pairwise = {}
    methods = list(METHOD_SHORT_NAMES.values())
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            overlap = len(miss_sets[m1] & miss_sets[m2])
            union = len(miss_sets[m1] | miss_sets[m2])
            jaccard = overlap / union if union else float("nan")
            pairwise[f"{m1}_vs_{m2}"] = {
                "overlap_count": overlap,
                "jaccard": jaccard,
                f"{m1}_only": len(miss_sets[m1] - miss_sets[m2]),
                f"{m2}_only": len(miss_sets[m2] - miss_sets[m1]),
            }

    n_total = originals["family_id"].nunique()
    return {
        "n_total_originals": int(n_total),
        "miss_count_histogram": histogram,
        "n_missed_by_all_6": histogram.get(6, 0),
        "n_never_missed": histogram.get(0, 0),
        "per_method_miss_count": {m: len(s) for m, s in miss_sets.items()},
        "pairwise_overlap": pairwise,
    }


def main() -> None:
    family_table = load_family_table("test")
    wide = build_wide_predictions(family_table)
    sia = _sentiment_analyzer()

    out_dir = sign_settings.results_dir / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    fn = build_false_negatives(wide, sia)
    fn.to_csv(out_dir / "false_negatives.csv", index=False, encoding="utf-8-sig")
    print(f"false_negatives.csv: {len(fn)} rows (originals missed by >=1 method)")

    fp = build_false_positives(wide, sia)
    fp.to_csv(out_dir / "false_positives.csv", index=False, encoding="utf-8-sig")
    print(f"false_positives.csv: {len(fp)} rows (interpretations flagged by >=1 method)")

    contrast = quantitative_contrast(fn, wide, sia)
    with open(out_dir / "quantitative_contrast.json", "w", encoding="utf-8") as f:
        json.dump(contrast, f, indent=2, ensure_ascii=False)
    print(f"quantitative_contrast.json: {contrast['n_ever_missed']} ever-missed vs {contrast['n_always_detected']} always-detected")

    overlap = cross_model_overlap(wide)
    with open(out_dir / "cross_model_overlap.json", "w", encoding="utf-8") as f:
        json.dump(overlap, f, indent=2, ensure_ascii=False)
    print(f"cross_model_overlap.json: {overlap['n_missed_by_all_6']} missed by all 6, {overlap['n_never_missed']} never missed")


if __name__ == "__main__":
    main()
