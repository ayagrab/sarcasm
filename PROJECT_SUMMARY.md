# Sarcasm Detection — Project Summary

Clean, cumulative, high-level document. For the operational/chronological
detail (exact commands run, blockers, environment audit) see
`EXPERIMENT_LOG.md`. This document is updated when meaningful conclusions
are available; unmeasured results are marked `TBD`, never guessed.

**Status (2026-08-12):** M1 (EXP-001) and M2 (EXP-002) are DEV-evaluated
(TEST sealed, per policy below). The Azure VM was restarted and its
ephemeral `/mnt` disk was wiped in the process (repo checkout, Python env,
model cache, and EXP-003's raw predictions were all lost, but not EXP-003's
metrics, which had already been recorded here and in `EXPERIMENT_LOG.md`);
the environment was fully recreated with the identical pinned stack and
verified, and all Stage A/B work is now committed to git rather than living
only on the VM's ephemeral disk or the local Mac's uncommitted working
tree. Full incident/recovery detail: `EXPERIMENT_LOG.md`, "VM restart --
`/mnt` ephemeral-disk data loss incident and recovery." M3 (EXP-003/004)
and M4 (EXP-005) are running now; see `STAGE_B_CHECKLIST.md` for the live
checklist.

**Web demo (`web/`, in progress in parallel with Stage B):** a FastAPI +
Next.js app (Simple Mode + Research/comparison Mode) that consumes this
project's classification code through inference adapters -- built and
tested (backend: 17/17 tests passing, TF-IDF path exercised for real;
frontend: builds cleanly, all three pages verified live against the
backend). It intentionally never re-tunes anything against sentences
typed into the UI, and gates every method except the already-frozen M1
(TF-IDF) behind Stage B's Phase 2 freeze -- so right now Simple Mode
serves TF-IDF and Research Mode shows the rest as "not frozen yet" /
"not trained yet," honestly, not as placeholders. See `web/README.md`.

## 1. Problem Definition

Given a short English text (a forum post / tweet-length message), predict
whether it is:

- `sarcastic`
- `not_sarcastic`

This is a **new phase** of the `sarcasm` repository, distinct from the
repository's existing (and already-implemented) work, which is a *sarcasm
interpretation/neutralization* benchmark (rewriting a known-sarcastic tweet
into a sincere sentence, then judging the rewrite — see the repo's root
`README.md`). That existing pipeline does not classify sarcasm; this phase
does. See `EXPERIMENT_LOG.md` → "Relationship to the existing repository"
for the full detail.

The goal here is not just to train one classifier, but to run a fair,
reproducible comparison across fundamentally different approaches, and
determine which is best under matched evaluation conditions — and why.

## 2. Dataset

**Sarcasm Corpus V2** (UC Santa Cruz), already staged in this repository at
`data/raw/sarcasm_corpus_v2/` (untouched, read-only). Three category
subsets, each independently ~50/50 balanced:

| Category | Meaning | Rows | sarc / notsarc |
|---|---|---:|---|
| GEN | General Sarcasm | 6,520 | 3,260 / 3,260 |
| HYP | Hyperbole | 1,164 | 582 / 582 |
| RQ | Rhetorical Questions | 1,702 | 851 / 851 |
| **Total** | | **9,386** | **4,693 / 4,693** |

No missing values. No within-file exact duplicates. **336 rows (168
groups) share exact text across category files** (mostly GEN↔RQ, since RQ
and HYP posts are sarcasm sub-phenomena also present in the general
corpus); of those, **4 groups (8 rows) carry conflicting labels** between
categories — a genuine annotation inconsistency in the source data, kept
and flagged rather than dropped. No author/conversation/timestamp metadata
exists in the raw files, so the only leakage vector identified is
duplicate/near-duplicate text — handled via grouped splitting (below). Full
detail: `EXPERIMENT_LOG.md` → "Dataset Information".

## 3. Experimental Methodology

- **Canonical dataset**: all three category files combined into one table
  with a global `example_id`, `category`, `source_file`, `dup_group_id`,
  and `label_conflict` flag. Non-destructive — no row is dropped.
  (`src/classification/data/build_canonical_dataset.py`)
- **Canonical split**: one fixed 70/15/15 train/dev/test split (seed 42),
  grouped by normalized-text duplicate group (so the same underlying post
  never appears in two different splits) and stratified by label at the
  group level. Persisted to `data/splits/` and reused, unmodified, by
  every approach below. (`src/classification/data/make_splits.py`)
- **Shared evaluator**: one implementation of Accuracy, per-class
  Precision/Recall/F1, Macro F1 (primary model-selection metric), Weighted
  F1, and confusion matrix, plus per-example prediction storage for later
  error analysis, used identically by every approach.
  (`src/classification/evaluation/metrics.py`)
- **Test-set discipline**: the test split is only touched for final,
  frozen-configuration evaluation — never for prompt iteration, few-shot
  selection, DSPy optimization, or hyperparameter tuning. See
  `EXPERIMENT_LOG.md` for the audit trail proving this per experiment.

## 4. Approaches Evaluated

| ID | Approach | Status |
|---|---|---|
| M1 | TF-IDF + Logistic Regression (classical baseline) | **EVALUATED** (EXP-001) |
| M2 | LLM zero-shot | Implemented, not yet run (blocked on API key) |
| M3 | LLM few-shot (random + curated variants) | Implemented, not yet run (blocked on API key) |
| M4 | LLM structured reasoning | Implemented, not yet run (blocked on API key) |
| M5 | DSPy-optimized LLM (Predict / BootstrapFewShot / MIPROv2) | Implemented, not yet run (blocked on API key + `dspy` install) |
| M6 | Fine-tuned English Transformer encoder | Implemented, not yet run (Stage B: needs model download) |

All LLM approaches (M2–M5) use the same underlying base LLM wherever
possible, to isolate the effect of prompting/optimization technique rather
than measuring base-model differences.

## 5. Evaluation Metrics

Accuracy, per-class (sarcastic / not-sarcastic) Precision/Recall/F1, Macro
F1 (primary), Weighted F1, confusion matrix. Same implementation for every
approach; see `src/classification/evaluation/metrics.py`.

## 6. Results

| Model | Accuracy | Macro F1 | Sarcastic P | Sarcastic R | Sarcastic F1 | Cost | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR (M1) | 0.740 | 0.740 | 0.724 | 0.758 | 0.741 | $0 | negligible (CPU, seconds) |
| LLM Zero-shot (M2) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| LLM Few-shot (M3) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| LLM Reasoning (M4) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DSPy (M5) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Fine-tuned Transformer (M6) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

All numbers above are on the frozen canonical **test** split (1,340
examples), computed by `src/classification/evaluation/metrics.py`. Full
detail (dev-set configuration sweep, confusion matrix, artifact paths) for
each row is in `EXPERIMENT_LOG.md`'s Experiment Registry. Remaining rows
are `TBD` until each approach is actually run (Stage B for M2-M6 — see
"Blockers" in `EXPERIMENT_LOG.md`).

## 7. Comparison Between Approaches

`TBD`.

## 8. Error Analysis

`TBD`.

## 9. Conclusions

`TBD`.

## 10. Limitations

- Sarcasm Corpus V2 is forum/social-media text from ~2016; may not
  represent contemporary sarcasm styles (slang, emoji-heavy text, etc.).
- No author/conversation metadata exists in the source data, so leakage
  control is limited to text-level deduplication/grouping — it's possible
  (though unlikely, given the observed duplicate patterns) that stylistic
  leakage from the same author appears elsewhere in the corpus without a
  detectable textual signature.
- 4 label-conflict groups (8 rows) represent inherent annotation
  disagreement in the source corpus; they're kept rather than resolved, so
  a small amount of label noise is expected to persist through every split.
- Development environment for this phase has no CUDA GPU (Apple MPS only)
  and limited free disk (~11 GiB) — see `EXPERIMENT_LOG.md` Blockers.

## 11. Recommended Production Approach

`TBD` — pending results.

## 12. Future Work

`TBD` — pending results; will include e.g. calibration analysis, ensemble
approaches, or category-specific (GEN/HYP/RQ) modeling if the results
motivate it.

## 13. Reproducing the Experiments

See `EXPERIMENT_LOG.md`'s Experiment Registry and "Stage A Readiness
Report" for the exact, currently-valid per-experiment commands. General
shape:

```bash
# 0. Install dependencies (base + this phase's extras)
pip install -r requirements.txt -r requirements-classification.txt

# 1. Build the canonical dataset (from data/raw/sarcasm_corpus_v2/, read-only)
python -m src.classification.data.build_canonical_dataset

# 2. Audit it
python -m src.classification.data.audit_dataset

# 3. Build the canonical split (persisted under data/splits/)
python -m src.classification.data.make_splits

# 4. Run any experiment via its config file (see configs/, one per approach)
python -m src.classification.run_experiment --config configs/tfidf.json
python -m src.classification.run_experiment --config configs/llm_zero_shot.json
python -m src.classification.run_experiment --config configs/dspy_predict.json
python -m src.classification.run_experiment --config configs/transformer_roberta_base.json
```

Every experiment's configuration, metrics, and per-example predictions are
saved under `results/<experiment_id>/`. LLM/DSPy experiments require a
real `OPENROUTER_API_KEY` in `.env`; DSPy additionally requires
`pip install dspy`; Transformer fine-tuning downloads its checkpoint from
Hugging Face on first run. None of these three have been executed in this
environment yet — see `EXPERIMENT_LOG.md` Blockers.

Run the test suite (never calls a real API or downloads a model):

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
