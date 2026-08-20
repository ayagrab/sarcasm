"""Phase 2 (SIGN_GENERALIZATION_PLAN.md): text-level characterization
stats for one corpus (Dataset A / SIGN originals / SIGN interpretations),
computed independently per corpus so they can be compared side by side.
No cross-corpus logic lives here -- see `run_characterization.py` for the
three-way comparison and write-up.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from src.sign.characterization.nltk_setup import ensure_nltk_resources

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "as",
    "by", "that", "this", "it", "i", "you", "he", "she", "we", "they",
    "my", "your", "his", "her", "its", "our", "their", "not", "no", "do",
    "does", "did", "have", "has", "had", "if", "so", "just", "there",
    "all", "will", "would", "can", "could", "should", "what", "which",
    "who", "when", "where", "how", "why", "am",
}


def tokenize(text: str) -> list[str]:
    ensure_nltk_resources()
    from nltk.tokenize import word_tokenize

    return word_tokenize(str(text))


def _word_tokens_only(tokens: list[str]) -> list[str]:
    return [t.lower() for t in tokens if re.search(r"[A-Za-z]", t)]


def length_stats(texts: pd.Series) -> dict:
    char_lens = texts.astype(str).str.len()
    word_lens = texts.astype(str).str.split().apply(len)
    return {
        "char_length": {
            "mean": float(char_lens.mean()),
            "median": float(char_lens.median()),
            "std": float(char_lens.std()),
            "min": int(char_lens.min()),
            "max": int(char_lens.max()),
        },
        "word_length": {
            "mean": float(word_lens.mean()),
            "median": float(word_lens.median()),
            "std": float(word_lens.std()),
            "min": int(word_lens.min()),
            "max": int(word_lens.max()),
        },
    }


def vocabulary_stats(texts: pd.Series, sample_n: int | None = 3000, seed: int = 42) -> dict:
    """Vocabulary size + type-token ratio (lexical diversity). TTR is
    computed on a fixed-size random sample (default 3,000 texts) when the
    corpus is larger, so corpora of very different sizes (Dataset A's
    9,386 vs. SIGN interpretations' ~15,000) are compared on equal
    footing -- TTR is well known to shrink mechanically as corpus size
    grows, so comparing raw-size TTRs would be comparing sample size, not
    lexical diversity."""
    all_tokens: list[str] = []
    for t in texts.astype(str):
        all_tokens.extend(_word_tokens_only(tokenize(t)))
    vocab_size = len(set(all_tokens))

    sample_texts = texts if sample_n is None or len(texts) <= sample_n else texts.sample(sample_n, random_state=seed)
    sample_tokens: list[str] = []
    for t in sample_texts.astype(str):
        sample_tokens.extend(_word_tokens_only(tokenize(t)))
    ttr = len(set(sample_tokens)) / len(sample_tokens) if sample_tokens else float("nan")

    return {
        "vocab_size": vocab_size,
        "total_word_tokens": len(all_tokens),
        "type_token_ratio_sampled": ttr,
        "ttr_sample_size": len(sample_texts),
    }


def top_ngrams(texts: pd.Series, n: int = 1, top_k: int = 20, exclude_stopwords: bool = True) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for t in texts.astype(str):
        words = _word_tokens_only(tokenize(t))
        if exclude_stopwords:
            words = [w for w in words if w not in _STOPWORDS]
        grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        counter.update(grams)
    return counter.most_common(top_k)


def punctuation_and_case_stats(texts: pd.Series) -> dict:
    s = texts.astype(str)
    return {
        "question_mark_rate": float(s.str.contains(r"\?").mean()),
        "exclamation_mark_rate": float(s.str.contains(r"!").mean()),
        "any_punctuation_rate": float(s.str.contains(r"[.!?,;:]").mean()),
        "ellipsis_rate": float(s.str.contains(r"\.\.\.").mean()),
        "quote_rate": float(s.str.contains(r'["“”]').mean()),
        "any_uppercase_char_rate": float(s.apply(lambda t: any(c.isupper() for c in t)).mean()),
        "any_fully_uppercase_word_rate": float(
            s.apply(lambda t: any(w.isupper() and len(w) > 1 for w in t.split())).mean()
        ),
        "mean_uppercase_char_fraction": float(
            s.apply(lambda t: (sum(1 for c in t if c.isupper()) / len(t)) if len(t) else 0.0).mean()
        ),
    }


def duplicate_stats(texts: pd.Series) -> dict:
    n = len(texts)
    exact_dupe_rate = 1.0 - (texts.nunique() / n) if n else float("nan")
    normalized = texts.astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    near_dupe_rate = 1.0 - (normalized.nunique() / n) if n else float("nan")
    return {
        "n_total": n,
        "n_unique_exact": int(texts.nunique()),
        "exact_duplicate_rate": float(exact_dupe_rate),
        "n_unique_normalized": int(normalized.nunique()),
        "near_duplicate_rate": float(near_dupe_rate),
    }


def sentiment_stats(texts: pd.Series, sample_n: int | None = 2000, seed: int = 42) -> dict:
    """Mean/std VADER compound sentiment score, on a fixed-size sample
    for consistency with `vocabulary_stats`' sampling rationale (VADER
    scoring is O(n) but comparing means across differently-sized samples
    is still fairer at matched sample size)."""
    ensure_nltk_resources()
    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    sample = texts if sample_n is None or len(texts) <= sample_n else texts.sample(sample_n, random_state=seed)
    scores = sample.astype(str).apply(lambda t: sia.polarity_scores(t)["compound"])
    return {
        "compound_mean": float(scores.mean()),
        "compound_std": float(scores.std()),
        "frac_positive_gt_0.3": float((scores > 0.3).mean()),
        "frac_negative_lt_neg0.3": float((scores < -0.3).mean()),
        "sample_size": len(sample),
    }


def describe_corpus(texts: pd.Series, seed: int = 42) -> dict:
    """All Phase-2 text-level stats for one corpus (a pandas Series of
    raw strings), in one call."""
    texts = texts.astype(str)
    return {
        "n_examples": len(texts),
        **length_stats(texts),
        "vocabulary": vocabulary_stats(texts, seed=seed),
        "top_unigrams": top_ngrams(texts, n=1, top_k=20),
        "top_bigrams": top_ngrams(texts, n=2, top_k=20),
        "punctuation_and_case": punctuation_and_case_stats(texts),
        "duplicates": duplicate_stats(texts),
        "sentiment": sentiment_stats(texts, seed=seed),
    }
