# SIGN Generalization Plan

**Master roadmap + live status for the new research phase.** This document
is the primary source of truth for this phase — a session that has only
this file plus `PROJECT_SUMMARY.md` / `EXPERIMENT_LOG.md` should be able to
understand the whole phase and resume it correctly. It will be updated
continuously (start/end of every stage, every blocker, every methodological
decision) — do not rely on chat history as the record of project state.

Created: 2026-08-20. Owner: ayagrab. Repo:
`https://github.com/ayagrab/sarcasm.git` (`main`).

---

## 0. What this project is (read this first)

The `sarcasm` repo has two **completed** parts, both frozen:

- **Part I — Interpretation.** Given a sarcastic tweet, an LLM rewrites it
  as a sincere sentence; the rewrite is scored by automatic metrics, an LLM
  judge, and human annotators (Alt-Test, Fleiss' Kappa, etc.). Source data:
  SIGN paper's original test data, **sarcastic tweets only, interpretations
  discarded** (`src/preprocessing/clean_dataset.py`). See `PROJECT_SUMMARY.md`
  Part I.
- **Part II — Detection ("Stage B").** Given a short text, predict
  sarcastic/not-sarcastic. Six methods (M1–M6) compared on **Dataset A**
  (defined below), each frozen to one config, each evaluated exactly once
  on a sealed TEST split. **Status: complete**, TEST scores frozen. See
  `PROJECT_SUMMARY.md` Part II and `EXPERIMENT_LOG.md`.

**This new phase (what this document plans) is a third, additive part:**
dataset comparison, cross-dataset generalization, and domain adaptation
between Dataset A and the **full** SIGN dataset (including the human
interpretations Part I discarded). It reuses Part II's six trained/frozen
methods as its starting point but does not touch their code, configs, or
results. Internally this phase is tracked as **"SIGN generalization"** or
**Part III**; every new experiment ID is prefixed `EXP-SIGN-###` (Part II
kept plain `EXP-0##`) precisely so the two studies never get confused in
`results/`, `EXPERIMENT_LOG.md`, or anywhere else.

### Hard constraints carried over from the user's brief (do not violate)

1. **Never modify** Part II's frozen configs, predictions, metrics, or
   `models/EXP-009/best_checkpoint` because of a SIGN result. New SIGN-tuned
   variants get new experiment IDs and new artifact paths, always.
2. **SIGN families must never leak.** The 5 interpretations of one
   sarcastic tweet are one family; never split a family across train/dev/
   test or across a training/evaluation boundary within an experiment.
3. **SIGN Test is evaluation-only, forever.** No training, prompt
   optimization, few-shot example selection, hyperparameter tuning, or
   checkpoint selection may look at SIGN Test. SIGN Train/Dev only for
   development.
4. **Zero-transfer first.** Every method's *un-adapted* performance on SIGN
   must be evaluated and persisted before that method is exposed to SIGN
   Train in any way (Phase 8 in `docs/finetuning_plan.md` numbering below
   is Phase 7/8 in this doc — see the phase table).
5. Note for the record: Part I already looked at (a subset of) SIGN's
   sarcastic tweets during dataset cleaning/generation. That exposure was
   to the **sarcastic originals only** (interpretations were stripped
   before Part I ever ran), for a *rewriting* task, not classification —
   it did not touch classifier training, prompt optimization, or model
   selection for detection. Disclosed here as the brief requires; judged
   not to invalidate the zero-transfer classification baseline.

---

## 1. The two datasets, precisely

### Dataset A — the ~9,386-sentence classification corpus (Part II's dataset)

Sarcasm Corpus V2 (UC Santa Cruz), canonicalized at
`data/processed/sarcasm_v2_canonical.csv`, split at `data/splits/`:

| Split | Rows | File |
|---|---:|---|
| Train | 6,706 | `data/splits/train.csv` |
| Dev | 1,340 | `data/splits/dev.csv` |
| Test (sealed) | 1,340 | `data/splits/test.csv` |
| **Total** | **9,386** | — |

Binary label (`not_sarcastic`/`sarcastic`), perfectly class-balanced.
**Authoritative and frozen — this phase only reads these files, never
regenerates or reassigns them.**

### Dataset B — SIGN

**Already present in this repo**, at `data/raw/original_{train,dev,test}_dataset.csv`
— no download needed. Format: **no header**, 2 columns per row
(`sarcastic_original, human_interpretation`), one row per
(original, interpretation) pair. This is the *pairs* view already staged
by an earlier stage of Part I; this phase adds a family-aware loader on
top of it (Phase/Stage 1 below) rather than re-deriving it.

**Verified row counts (2026-08-20 audit, stdlib `csv`, confirmed clean
2-field rows throughout — no embedded-comma corruption):**

| Split | Pairs (rows) | Official spec | Unique original texts | Official spec | "Clean" families (exactly 1 original + 5 unique interpretations) |
|---|---:|---:|---:|---:|---:|
| Train | 12,000 | 12,000 ✅ | 2,292 | 2,400 | 2,185 |
| Dev | 1,500 | 1,500 ✅ | 270 | 300 | 240 |
| Test | **1,470** | 1,500 ⚠️ | 265 | 300 | 237 |

**Known data-quality findings (must be documented, not silently
worked around):**

- **Test has 1,470 pairs, not the official 1,500** (30 short). This is in
  the raw file as committed, not something this audit introduced — it's
  the same file Part I already built on (see next point). Treated as a
  known limitation of the locally-available SIGN copy; not blocking, but
  disclosed everywhere Test counts are reported.
- **Family grouping only has text as a key — no tweet ID column exists in
  this file.** Grouping by exact stripped original text gives fewer
  unique families than the official spec (2,292 vs 2,400 train; 270 vs
  300 dev; 265 vs 300 test) and a nonzero number of "anomalous" families
  (107 train / 30 dev / 28 test) that don't have exactly 5 interpretation
  rows — almost certainly because a handful of distinct source tweets
  normalize to identical text (or were genuinely duplicated), merging two
  families under one text key. **Decision:** the family-aware loader
  (Phase/Stage 1) computes family membership by exact original-text match
  (the only key the data supports), and experiments that require exactly
  5 interpretations per family (Phase 5 strict/soft family metrics, Phase
  7/9/10 sampling) use only the **clean** (exactly-5) subset, with the
  excluded/anomalous family count reported alongside every such result.
  Loader-level tests assert these counts don't silently drift.
- **This "265 not 300" test-set shrinkage is corroborated by prior,
  independent work**: `data/processed/clean_sarcastic_sentences.csv`
  (Part I's already-completed dedup of the same `original_test_dataset.csv`
  file, `src/preprocessing/clean_dataset.py`) contains exactly **265**
  unique sentences — the same number this audit independently derived.
  Good cross-check that the discrepancy is a property of the source data,
  not a bug in either script.
- 944/2,292 train families (and proportionally in dev/test) contain at
  least one pair of byte-identical interpretation texts within the same
  family (different human annotators independently wrote the same
  rewrite, or a data artifact). Not a blocker for classification (labels
  are still correct) but noted for Phase 7's "1 interpretation per
  original" sampling, which should not be surprised by duplicate rows.
- Every sarcastic original in SIGN is, by construction, `sarcastic`; every
  interpretation is `not_sarcastic`. SIGN is **never** to be described as
  "15,000 independent sarcastic examples" anywhere in this phase's
  documentation — it is 3,000 originals + 15,000 interpretations
  (nominally; 14,970 actually present, see above), organized in families.

### Reference material already in the repo

- `docs/sign_paper.pdf` — the SIGN paper itself.
- `docs/methodology.md` §1, `docs/project_history.md` §"Starting point: the
  original SIGN paper" — how Part I used SIGN's sarcastic-only subset.

---

## 2. Research questions

- **RQ1** (already answered by Part II, restated here as the anchor):
  which classification approach performs best on Dataset A? → **M6
  (fine-tuned DeBERTa-v3-base), TEST Macro F1 0.8209.** Not reopened by
  this phase.
- **RQ2:** how well do Dataset-A-developed classifiers generalize to the
  structurally different SIGN corpus, zero-transfer?
- **RQ3:** how much SIGN Train data is needed to close the gap found in
  RQ2?
- **RQ4** (secondary): does using more of SIGN's 5 human interpretations
  per tweet (vs. just 1) improve the classifier's learned boundary?

Phases 1–3 below build the evidence base (is there even a domain shift to
explain?); Phases 4–6 answer RQ2 and characterize *why*; Phases 7–10
answer RQ3/RQ4; Phase 11 synthesizes all of it.

---

## 3. Scope decision: which of M1–M6 get full adaptation treatment

Flagging this up front as a **methodological decision**, not hidden in the
weeds — happy to be redirected.

- **Phases 1–6 (characterization, origin classification, zero-transfer,
  family eval, error analysis) run for all six methods where technically
  meaningful.** M1 and M6 are genuine trained classifiers; M2–M4 are
  prompted (no training, so "zero-transfer" is just "run the frozen
  prompt on new data" — cheap); M5 is a frozen, already-compiled DSPy
  program (zero-transfer = inference only, not recompilation — cheap).
- **Phases 7–10 (SIGN Train prep, domain adaptation, learning curves,
  interpretation-count ablation) are scoped to M1 and M6 only.**
  Rationale: these phases are inherently about *retraining on new
  labeled data*, which only cleanly applies to a trained classifier. M1
  is free to iterate (CPU, seconds/run) and is the natural workhorse for
  the learning-curve/ablation grids (many runs). M6 is the project's best
  model and the one a real domain-adaptation conclusion should be about;
  its retraining is VM-bound but each run is cheap (~22 min for a
  Dataset-A-sized run, Part II EXP-009). M2–M4 have no training step to
  adapt (adapting them would mean redesigning the few-shot examples or
  prompt around SIGN, a different research question); M5 *could* be
  re-compiled with SIGN-derived examples but at ~1.5–2.8h per compile
  (Part II M5 numbers) for a method that already trails M1/M6 by a wide
  margin, that cost isn't well spent unless requested.
- If this scoping is wrong for the report's needs (e.g. a supervisor
  wants M5 adapted too), it's a cheap addition later — Phase 7's SIGN
  Train variants are method-agnostic and would already exist.

---

## 4. Phase-by-phase plan

Each phase below carries the fields the brief requires. **Status** starts
`NOT STARTED` for everything except Phase 0 (this document) and the
Phase-1(-of-the-brief)/audit work folded into Phase 0.

### Phase 0 — Audit & Planning *(this document)*

- **Goal:** understand existing repo state, verify SIGN's schema/counts/
  family structure, produce this plan.
- **Status:** **COMPLETED** 2026-08-20.
- **VM required:** NO.
- **What was done:** full audit of `README.md`, `PROJECT_SUMMARY.md`,
  `EXPERIMENT_LOG.md`, `configs/`, `results/`, `models/`, `data/`,
  `config/`, `src/classification/`; verified Part II is complete and
  frozen (all M1–M6 TEST results present, `results/EXP-00{1..9}(-TEST)/`);
  verified SIGN's official splits are already present at
  `data/raw/original_{train,dev,test}_dataset.csv`; ran a real family-
  structure audit (counts in §1 above); confirmed `models/EXP-009/best_checkpoint`
  (M6) is available locally for CPU inference; confirmed the Qwen local
  client (`src/classification/llm/local_client.py`) hard-requires CUDA, so
  M2–M5 need the Azure VM; confirmed the VM-only sync/verify scripts were
  deliberately removed from the repo after Part II finished (commit
  `7a17e8e`) — recoverable via `git show 7a17e8e~1:scripts/<name>` when
  needed; found and fixed a broken local `.venv` (stale reference to a
  deleted `.venv_arm64` interpreter — recreated with `/usr/local/bin/python3.10`,
  reinstalled `requirements.txt` + `requirements-classification.txt` +
  `requirements-dev.txt`, verified importable).
- **Problems found:** SIGN Test row-count discrepancy (1,470 vs. spec's
  1,500) and family-grouping-by-text imprecision — both documented in §1,
  neither blocking.
- **Artifacts:** this document; no data/code artifacts yet.
- **Next action:** Phase 1 (Foundation).

### Phase 1 — Foundation: SIGN loaders, family grouping, leakage tests

*(Corresponds to "Implementation strategy" items 1–4 in the brief.)*

- **Research question:** none directly — infrastructure for every later
  phase.
- **Inputs:** `data/raw/original_{train,dev,test}_dataset.csv` (read-only,
  untouched).
- **Status:** **COMPLETED** 2026-08-20.
- **VM required:** NO. Ran entirely locally.
- **What was built:**
  - `config/sign_settings.py` — Part III's settings (mirrors
    `config/classification_settings.py`'s pattern), incl. the label
    convention (`sarcastic`/`not_sarcastic`, matching Dataset A 1:1, no
    remapping needed anywhere downstream) and `random_seed = 42`.
  - `src/sign/data/load_sign.py` — `load_raw_pairs` (stdlib `csv`, not
    `pandas`, so the header-less files can't be silently misparsed;
    verifies every row has exactly 2 fields), `build_family_table`
    (groups by exact stripped original text, in first-appearance order,
    into `family_id = f"{split}-{i:05d}"`), `summarize`, and a CLI
    (`python -m src.sign.data.load_sign --rebuild`) that writes
    `data/sign/family_table_{split}.csv`.
  - `src/sign/data/family_utils.py` — `assert_no_family_leakage`,
    deterministic `sample_family_ids` (seeded, sorted-input, frac-or-n),
    `select_families` (whole families only), `clean_families_only`,
    `select_k_interpretations_per_family` (per-family seeded shuffle,
    so k=1/2/3/5 selections are **nested subsets** — controls Phase 10's
    ablation so it isn't confounded by which interpretations are used),
    `to_classification_frame` (flattens to the `example_id/text/gold_label`
    schema `src.classification.evaluation.metrics` already expects).
  - `src/sign/family_eval/metrics.py` — pulled forward from Phase 5 since
    it's pure computation with no model dependency: `original_sarcasm_detection_rate`,
    `interpretation_non_sarcasm_rate`, `pairwise_contrastive_accuracy`,
    `strict_family_accuracy`, `soft_family_score`, `compute_family_metrics`.
    Fully spec'd and tested now; Phase 5 will just call it on real
    predictions.
  - **Data artifacts:** `data/sign/family_table_{train,dev,test}.csv` —
    durable, long-format, family-labeled SIGN tables; every later phase
    reads these instead of re-parsing the raw files.
  - **Tests (all passing, 37 new / 187 total repo-wide, zero
    regressions):** `tests/test_sign_parsing.py`,
    `tests/test_sign_family_grouping.py`, `tests/test_sign_labels.py`,
    `tests/test_sign_split_integrity.py` (locks in the real, audited
    counts below as a regression guard), `tests/test_sign_sampling.py`,
    `tests/test_sign_no_leakage.py`, `tests/test_sign_family_metrics.py`.
  - `configs/sign/` and `results/sign/` directories created (empty,
    ready for Phase 3+).
- **Verified real counts (built table == hand-audit in §1, exactly):**
  train 14,292 rows / 2,292 families / 2,185 clean; dev 1,770 / 270 / 240;
  test 1,735 / 265 / 237.
- **Problems encountered:** none new (the local `.venv` fix from Phase 0
  was a prerequisite, not a Phase 1 problem).
- **Next action:** Phase 2 (dataset characterization) — can start
  immediately, local-only, no VM.

### Phase 2 — Dataset characterization (Dataset A vs. SIGN)

- **Research question:** how different are the two datasets; is there
  visible domain shift?
- **Inputs:** Dataset A splits (read-only) + SIGN loader output (Phase 1).
- **Outputs:** `results/sign/characterization/` — summary tables (length,
  vocab, lexical diversity, n-grams, punctuation, sentiment, duplicates)
  computed **separately** for Dataset A, SIGN originals, and SIGN
  interpretations (three-way, never SIGN-as-one-blob); sentence-embedding
  PCA/UMAP plots (`sentence-transformers`, CPU-fine for ~25k short texts);
  a written findings section appended to this document (§6 below) and to
  `EXPERIMENT_LOG.md`.
- **New dependency needed:** `sentence-transformers` (+ `umap-learn` if
  UMAP is used over PCA-only) — not currently in
  `requirements-classification.txt`; will be added there when this phase
  starts.
- **Compute:** CPU embedding of ~25k short sentences with a small
  sentence-transformer model (e.g. `all-MiniLM-L6-v2`) — order of 10–20
  min on a laptop CPU; everything else (n-grams, PCA) is seconds.
- **VM required:** NO.
- **Estimated time:** ~2–3h (mostly plotting/writeup, not runtime).
  **Actual:** ~25 min including a real run (embedding 6,000 sampled
  sentences took well under a minute on CPU — the 10–20 min estimate was
  conservative).
- **Status:** **COMPLETED** 2026-08-20.
- **What was built:** `src/sign/characterization/stats.py` (length,
  vocabulary/TTR sampled at matched size, top n-grams, punctuation/case,
  exact+near-duplicate rates, VADER sentiment — nltk's `punkt_tab` /
  `vader_lexicon` needed a one-time download, blocked by this machine's
  default SSL context; fixed via `certifi`'s cert bundle in
  `nltk_setup.py`, cached locally afterward, no further network needed),
  `src/sign/characterization/embeddings.py` (`all-MiniLM-L6-v2` CPU
  embeddings + PCA + UMAP, equal-sized-per-group deterministic sampling
  so the plot isn't size-dominated by SIGN interpretations' larger
  count), `src/sign/characterization/run_characterization.py` (CLI
  orchestrator). Outputs: `results/sign/characterization/corpus_stats.json`,
  `embeddings_2d.csv`, and 5 PNGs under `figures/`.
- **Findings:** see §6 below.
- **Next action:** Phase 3 (origin classification) — informed directly by
  this phase's punctuation/capitalization finding (see §6).

### Phase 3 — Dataset-origin classification (diagnostic)

- **Research question:** are the two datasets trivially distinguishable
  by surface features? (Not a product classifier.)
- **Inputs:** Dataset A + SIGN (both roles, clearly labeled by origin, not
  by sarcasm label — this is a *dataset-of-origin* label, a third,
  separate axis).
- **Outputs:** `results/sign/EXP-SIGN-001/` — TF-IDF+LR origin classifier,
  accuracy/macro-F1/confusion matrix, methodology note (train/eval split
  built from Dataset A's own splits + SIGN's own splits, no leakage —
  e.g. train on {Dataset-A train, SIGN train}, eval on {Dataset-A test,
  SIGN test}).
- **Compute:** seconds (same class of job as Part II's M1).
- **VM required:** NO.
- **Estimated time:** ~1h including writeup. **Actual:** ~25 min.
- **Status:** **COMPLETED** 2026-08-20.
- **What was built:** `src/sign/origin_classification/run_origin_classifier.py`
  — reuses Part II's exact winning TF-IDF vectorizer config (word 1-2gram
  + char_wb 3-5gram `FeatureUnion`, imported from
  `src.classification.classical.tfidf_baseline`, not duplicated) +
  `LogisticRegression(class_weight="balanced")` (SIGN train outnumbers
  Dataset A train ~2:1). **Two conditions**, motivated directly by Phase
  2's punctuation/case finding: `EXP-SIGN-001` (raw text) and
  `EXP-SIGN-002` (case+punctuation-normalized text, both corpora, to
  isolate content signal from surface formatting). Train: Dataset A train
  (6,706) + SIGN train, all roles (14,292). Eval: Dataset A test (1,340) +
  SIGN test, all roles (1,735) — each corpus's own held-out test
  partition, nothing new touched.
- **Results:**

  | Experiment | Condition | Accuracy | Macro F1 | Confusion matrix (rows=gold, cols=pred; order dataset_a, sign) |
  |---|---|---:|---:|---|
  | EXP-SIGN-001 | raw text | 0.9561 | 0.9555 | `[[1298, 42], [93, 1642]]` |
  | EXP-SIGN-002 | case+punct normalized | 0.9242 | 0.9235 | `[[1267, 73], [160, 1575]]` |

- **Interpretation:** the two corpora are **very easily distinguishable**
  even *after* removing the punctuation/capitalization confound flagged
  in Phase 2 — macro F1 only drops 3.2 points (0.9555 → 0.9235) when
  surface formatting is stripped out. This means the separability isn't
  primarily a formatting artifact: genuine topical/lexical/length
  differences (Phase 2: forum-argument register vs. tweet-length
  personal-life text) are doing most of the work. **Strong, direct
  evidence of substantial domain shift**, going into Phase 4 as the
  expectation to test against (does this shift actually hurt sarcasm
  detection, or is it orthogonal to the task).
- **Next action:** Phase 4 (zero-transfer) — **requires the Azure VM**
  for M2–M5; see the checkpoint report for the explicit ask.

### Phase 4 — Zero-transfer to SIGN (all six methods, no SIGN Train exposure)

**This is the scientifically critical phase — nothing in Phase 7+ may
start until this is fully persisted.**

- **Research question:** RQ2 — how well do Dataset-A classifiers
  generalize to SIGN, untouched?
- **Inputs:** SIGN Test (all roles: 265 originals + their interpretations,
  clean-family subset flagged separately from the full set) — **SIGN Dev
  may also be evaluated for a secondary, non-selection-driving data
  point, SIGN Train is not used at all in this phase.**
- **Method-by-method plan:**

  | Method | What "zero-transfer eval" means | Compute | VM? |
  |---|---|---|---|
  | M1 (TF-IDF+LR) | Load frozen `configs/tfidf.json` model, predict on SIGN | seconds | NO |
  | M6 (DeBERTa) | Load `models/EXP-009/best_checkpoint`, `scripts/eval_frozen_checkpoint.py`-style eval, predict on SIGN | ~5–15 min on CPU (or seconds on VM GPU) | NO (CPU is fine; may batch onto VM anyway for convenience — see below) |
  | M2 (Qwen zero-shot) | Run frozen zero-shot prompt on SIGN via `LocalHFClient` | ~40–60 min (est. from Part II's 1.7s/example × ~1,800 SIGN Test rows) | **YES** |
  | M3 (Qwen few-shot) | Run frozen random-8 few-shot prompt on SIGN | ~1.5–2h (est.) | **YES** |
  | M4 (Qwen reasoning) | Run frozen structured-reasoning prompt on SIGN | ~1.3–1.5h (est.) | **YES** |
  | M5 (DSPy) | Run the **already-compiled, frozen** MIPROv2 program (`configs/dspy_mipro_v2.json`) in inference-only mode on SIGN — no recompilation | ~1–1.5h (est., inference only, not the 2h48m compile+eval Part II number) | **YES** |

- **Outputs per method:** `results/sign/EXP-SIGN-01{1..6}/` (or similar,
  finalized in Phase 1's ID scheme) with full predictions.csv +
  metrics.json, plus the specific brief-mandated numbers: count of the
  265–300 originals correctly predicted sarcastic, sarcasm recall on
  originals, false-negative rate.
- **VM required:** YES, for M2–M5 (M1/M6 done locally first, ahead of the
  VM session, to shorten VM time).
- **Estimated time:** ~4.5–6h of VM wall-clock for M2–M5 combined (first-
  run estimate — will be revised after M2 actually runs, per §5).
- **Safe to interrupt:** between methods YES; mid-method NO (no
  checkpointed resume for these inference runs, mirrors Part II's own
  runs). Each method's `predictions.csv` is written only on full
  completion — an interrupted run leaves no partial (and no incorrect)
  artifact.
- **Status:** NOT STARTED. **Depends on:** Phase 1. **Must complete and
  be persisted before Phase 7.**

### Phase 5 — SIGN contrastive / family-aware evaluation

- **Research question:** RQ2, at family granularity — not just "is each
  row right" but "does the model correctly separate an original from its
  own 5 rewrites."
- **Inputs:** Phase 4's predictions (no new model inference — this phase
  re-analyzes Phase 4's `predictions.csv` files with family-aware
  metrics). **No SIGN Train use.**
- **Outputs:** `results/sign/family_eval/` — per method: original
  detection rate, interpretation non-sarcasm rate, pairwise contrastive
  accuracy, strict family accuracy, soft family score; full per-family
  results table for qualitative follow-up (Phase 6).
- **Compute:** trivial (re-aggregating existing predictions).
- **VM required:** NO.
- **Estimated time:** ~2h.
- **Status:** NOT STARTED. **Depends on:** Phase 4.

### Phase 6 — Error analysis on SIGN

- **Research question:** RQ2's "why" — what kinds of sarcasm fail to
  transfer, and what are false-positive interpretations reacting to.
- **Inputs:** Phase 4/5 outputs (predictions + family metrics).
- **Outputs:** `results/sign/error_analysis/` — categorized false
  negatives (originals missed) and false positives (interpretations
  flagged sarcastic) with representative examples, cross-method
  comparison, qualitative write-up appended to this doc / `EXPERIMENT_LOG.md`.
- **Compute:** trivial; mostly manual/LLM-assisted qualitative review.
- **VM required:** NO.
- **Estimated time:** ~2–4h (qualitative work takes longer than it looks).
- **Status:** NOT STARTED. **Depends on:** Phase 5.

### Phase 7 — Prepare SIGN Train variants (data prep only, no training yet)

- **Research question:** none directly — controlled data prep for Phase
  8/9/10, gated on Phase 4 being safely persisted first (brief's hard
  requirement).
- **Inputs:** SIGN **Train** only (2,185 clean families after the Phase-1
  filter; 2,292 if anomalous families are kept for the non-strict
  variants — decided per-experiment and recorded).
- **Outputs:** `data/sign/train_variants/` — deterministic, seeded
  (`seed=42`, matching the rest of the project's convention) family-level
  samples: balanced (1:1, one interpretation/original), and 1/2/3/5-
  interpretations-per-original variants. Every variant file records its
  exact family_ids and the seed used; `assert_no_family_leakage` run on
  every variant as a saved check, not just at generation time.
- **Compute:** trivial (sampling ~2,185 families).
- **VM required:** NO.
- **Estimated time:** ~1–2h.
- **Status:** NOT STARTED. **Depends on:** Phase 4 (persisted) + Phase 1.

### Phase 8 — Domain adaptation (M1 + M6 only, see §3)

- **Research question:** RQ2/RQ3 — does SIGN Train exposure help, and
  does it hurt Dataset A performance (catastrophic forgetting check)?
- **Conditions ×2 models:** (A) Dataset-A-only [= Phase 4's zero-transfer,
  reused, not rerun], (B) Dataset A + SIGN Train, (C) SIGN Train only —
  each evaluated on SIGN Test **and** re-evaluated on Dataset A's own held-
  out TEST to check for forgetting.
- **Compute:** M1 — seconds/run, all local. M6 — ~15–30 min/run (scales
  with Part II's 22-min/6,706-example baseline), VM required.
- **VM required:** YES for M6's (B)/(C) runs (2 new fine-tunes; (A) is
  already-frozen `EXP-009`, reused not rerun).
- **Estimated time:** M1 ~30 min total. M6 ~1–1.5h VM time for both runs
  + eval.
- **Status:** NOT STARTED. **Depends on:** Phase 7, and Phase 4 fully
  persisted (hard gate).

### Phase 9 — Learning curve: how much SIGN Train is needed

- **Research question:** RQ3.
- **Inputs:** Phase 7's fractional family-level samples (0/10/25/50/75/100%
  of SIGN Train families), same SIGN Test throughout.
- **Outputs:** `results/sign/learning_curve/` — metrics (sarcasm recall on
  originals, interpretation accuracy, macro F1, pairwise contrastive
  accuracy, strict family accuracy) per fraction per model; plot of
  performance vs. fraction.
- **Compute:** M1 — 6 fractions × seconds, local, trivial. M6 — 6
  fractions × ~15–30 min ≈ 1.5–3h VM time (fewer examples at low
  fractions, cheaper).
- **VM required:** YES (M6 leg only).
- **Estimated time:** M1 ~1h incl. plotting. M6 ~2–3.5h VM time.
- **Status:** NOT STARTED. **Depends on:** Phase 7, Phase 4 persisted.

### Phase 10 — Interpretation-count ablation (1 vs 2 vs 3 vs 5 per tweet)

- **Research question:** RQ4.
- **Inputs:** Phase 7's 1/2/3/5-interpretations-per-original variants,
  with total example count and class balance controlled so the
  comparison isn't confounded by dataset size (documented explicitly per
  the brief — likely: fix total non-sarcastic examples via
  duplication/weighting when using fewer interpretations, or fix
  #families and let size vary but report both a size-matched and a
  natural-size condition).
- **Outputs:** `results/sign/interp_count_ablation/` — same metric set as
  Phase 9, compared across the 4 interpretation-count conditions.
- **Compute:** M1 — trivial, local. M6 — 4 conditions × ~15–30 min ≈
  1–2h VM time.
- **VM required:** YES (M6 leg only).
- **Estimated time:** M1 ~45 min. M6 ~1–2h VM time.
- **Status:** NOT STARTED. **Depends on:** Phase 7, Phase 4 persisted.

### Phase 11 — Final synthesis

- **Research question:** RQ1–RQ4, connected end-to-end.
- **Outputs:** a new "Part III — SIGN Generalization" section in
  `PROJECT_SUMMARY.md` (clean narrative + final tables), a completed
  `EXPERIMENT_LOG.md` SIGN section (full audit trail, every `EXP-SIGN-###`),
  this document marked fully COMPLETED.
- **VM required:** NO.
- **Estimated time:** ~2–4h writing/assembly, after all prior phases.
- **Status:** NOT STARTED. **Depends on:** everything above.

---

## 5. Time estimates — summary

| Phase | Local/dev time | VM time | VM required |
|---|---:|---:|:---:|
| 0 — Audit & plan | done | — | NO |
| 1 — Foundation | 2–3h | — | NO |
| 2 — Characterization | 2–3h | — | NO |
| 3 — Origin classifier | 1h | — | NO |
| 4 — Zero-transfer | ~1h (M1/M6 local prep) | 4.5–6h (M2–M5) | YES |
| 5 — Family eval | 2h | — | NO |
| 6 — Error analysis | 2–4h | — | NO |
| 7 — SIGN Train prep | 1–2h | — | NO |
| 8 — Domain adaptation | 0.5h (M1) | 1–1.5h (M6) | YES |
| 9 — Learning curve | 1h (M1) | 2–3.5h (M6) | YES |
| 10 — Interp-count ablation | 0.75h (M1) | 1–2h (M6) | YES |
| 11 — Synthesis | 2–4h | — | NO |
| **Total** | **~16–24h local/dev** | **~8.5–13h VM** | — |

These are **first-pass estimates** with real uncertainty (flagged per the
brief): the VM figures for M2–M5 (Phase 4) are extrapolated from Part
II's per-example rates on a differently-sized SIGN Test set and have not
been measured yet; M6 retraining times scale with dataset size in a way
that's only loosely linear. **Will be revised with real numbers after the
first VM session (Phase 4) — see the "revise estimates" note in §7.**

VM sessions can be **batched**: Phases 4, 8, 9, and 10 all need the VM,
but 8/9/10 depend on Phase 4 being *persisted* first (not on the VM being
restarted) — so the practical plan is likely **two VM sessions**: one for
Phase 4 (zero-transfer, must be isolated and persisted before anything
else touches SIGN Train), and one later batched session covering Phase 8
+ 9 + 10 together (all M6 retraining, ~4.5–7h combined) once Phase 7's
data prep is ready. This will be proposed explicitly when Phase 4 is
about to start.

---

## 6. Dataset characterization findings

*(Phase 2, run 2026-08-20. Full numbers: `results/sign/characterization/corpus_stats.json`.
Three corpora compared throughout, never merged: Dataset A [9,386,
train+dev+test], SIGN originals [2,827, all `sarcastic`], SIGN
interpretations [14,970, all `not_sarcastic`].)*

**Class structure.** Dataset A is 50/50 balanced by construction.
SIGN originals are 100% `sarcastic` (n=2,827); SIGN interpretations are
100% `not_sarcastic` (n=14,970) — confirmed mechanically, not just
asserted; SIGN is at no point a "15,000 independent sarcastic examples"
set.

**Length — the two datasets are structurally different genres.**
Dataset A averages **48.7 words/example** (median 38, up to 150) —
forum-post-length text. SIGN originals average **13.9 words** (median
13), interpretations **11.6 words** (median 10) — tweet-length text,
~3.5x shorter. Distributions barely overlap (`figures/length_distribution.png`).

**Punctuation and capitalization — the single largest, and most
methodologically important, surface difference.** Dataset A: 99.6% of
examples contain punctuation, 75.6% contain any uppercase character.
SIGN (both originals and interpretations): **~0.1–0.25% contain any
uppercase character, ~6–12% contain any punctuation at all.** SIGN's raw
text in this repo has clearly been lowercased and largely stripped of
punctuation upstream of this project (`figures/punctuation_case_comparison.png`).
**This is flagged as a methodological risk for Phase 3**: a dataset-of-
origin classifier could reach very high accuracy purely by detecting
"has an uppercase letter" / "has a period," which would say nothing
about topical/content differences. Phase 3's plan is updated (below) to
include a case/punctuation-normalized control condition so origin-
classification accuracy can be decomposed into "surface formatting" vs.
"content" — otherwise the diagnostic would answer the wrong question.

**Vocabulary.** Raw vocabulary size scales with corpus size as expected
(Dataset A 23,899 types over 466,726 tokens; SIGN interpretations 8,399
types over 175,394 tokens; SIGN originals 6,518 types over 39,522
tokens) — not comparable directly. At a **matched sample size (3,000
texts)**, type-token ratio is actually *higher* for SIGN (originals
0.165, interpretations 0.144) than Dataset A (0.088) — Dataset A's
longer examples repeat more function words per example, mechanically
lowering per-sample TTR; not read as "SIGN is more diverse" without that
caveat.

**Sentiment (VADER compound, matched 2,000-example samples) — a clean,
theoretically-expected split.** Dataset A: mean **+0.017** (near-neutral,
37.3% positive / 34.4% negative — a balanced discussion-forum corpus).
SIGN originals: mean **+0.27** (57.1% scored positive) — sarcastic
tweets surface-encode positive sentiment ("love," "great" are top-10
content words) which is exactly the *ironic* surface positivity sarcasm
is built on. SIGN interpretations: mean **−0.17** (45.4% scored
negative) — "hate" is the single most frequent content word (1,595
occurrences) and "bad" is top-10; interpretations spell out the
sarcastic tweet's *real*, usually negative, meaning. This
originals-positive / interpretations-negative sentiment reversal is
SIGN's structure showing up directly in a generic off-the-shelf
sentiment tool, and is a good independent sanity check that the family
labels are semantically meaningful, not just a formatting artifact.

**Duplicates.** Dataset A: 1.8% exact / 3.2% near-duplicate rate
(consistent with `dup_group_id` already tracked elsewhere in the
project). SIGN originals: negligible (0.07%) — each source tweet is
essentially unique. SIGN interpretations: **17.2%** exact-duplicate rate
— many human annotators converge on the same short, generic rewrite
("i hate this," "this is bad") across *different* source tweets. Not a
family-level artifact (that was already measured in Phase 1 — duplicate
interpretations *within* one family); this is duplication *across*
different families, a separate and expected property of short, generic
sincere rewrites.

**Embeddings (PCA/UMAP, `all-MiniLM-L6-v2`, 2,000/group equal sample,
`figures/embedding_pca.png` / `embedding_umap.png`).** Dataset A occupies
a visually distinct region of the projection, largely separated from
both SIGN roles. **SIGN originals and SIGN interpretations overlap
heavily with each other** in embedding space — far less separable from
each other than either is from Dataset A. Read together with Part II's
own headline finding (LLM methods systematically over-predict
"sarcastic," suggesting sarcasm isn't well captured by surface/semantic
similarity alone), this is a first piece of evidence, before any
classifier is even run, that **distinguishing SIGN originals from their
own interpretations is a harder, more fine-grained problem than
distinguishing SIGN from Dataset A at all** — motivating Phase 4's
zero-transfer test directly.

**Bottom line: yes, substantial domain shift exists between Dataset A
and SIGN** — genre (forum debate vs. tweet), length, formatting
(punctuation/casing), and topic/register (argumentative vs.
personal-life) all differ sharply. Phase 3 will quantify how easily a
simple classifier detects this; Phase 4 will show whether it hurts
sarcasm detection specifically.

## 7. Zero-transfer results

*(Populated after Phase 4 runs. Empty for now — this section, once
filled, is the phase's most important scientific artifact and must never
be overwritten by a later adaptation result; adaptation results get their
own §8/§9/§10 sections.)*

## 8. Domain adaptation results

*(Populated after Phase 8 runs.)*

## 9. Learning curve results

*(Populated after Phase 9 runs.)*

## 10. Interpretation-count ablation results

*(Populated after Phase 10 runs.)*

## 11. Final synthesis

*(Populated after Phase 11.)*

---

## 12. Checklist

- [x] Phase 0 — Audit & planning
- [x] Phase 1 — Foundation (loaders, family grouping, leakage tests)
- [x] Phase 2 — Dataset characterization
- [x] Phase 3 — Dataset-origin classification
- [ ] Phase 4 — Zero-transfer to SIGN (M1–M6) — **hard gate before Phase 7+**
- [ ] Phase 5 — SIGN contrastive/family evaluation
- [ ] Phase 6 — Error analysis
- [ ] Phase 7 — Prepare SIGN Train variants
- [ ] Phase 8 — Domain adaptation (M1 + M6)
- [ ] Phase 9 — Learning curve (M1 + M6)
- [ ] Phase 10 — Interpretation-count ablation (M1 + M6)
- [ ] Phase 11 — Final synthesis

Status legend: NOT STARTED / IN PROGRESS / COMPLETED / BLOCKED / WAITING
FOR VM / FAILED / SKIPPED.

---

## 13. What requires the VM vs. what's local-only

**Local-only (Mac, no VM):** Phases 0, 1, 2, 3, 5, 6, 7, 11; M1 legs of
Phases 4/8/9/10; M6's *zero-transfer inference* in Phase 4 (CPU-feasible
via the existing `models/EXP-009/best_checkpoint`, though it may be run
on the VM anyway for convenience if a VM session is open for M2–M5 at the
same time).

**VM required:** M2/M3/M4/M5 legs of Phase 4 (Qwen local inference and
DSPy program inference both hard-require CUDA); M6 legs of Phases 8/9/10
(fine-tuning).

**Before any VM stage:** the VM-only sync/verify tooling
(`scripts/sync_to_vm.sh`, `sync_from_vm.sh`, `verify_gpu.py`,
`verify_kernel.sh`) was deliberately removed from the repo after Part II
finished (commit `7a17e8e`). It will be restored from git history
(`git show 7a17e8e~1:scripts/<name>`) rather than rewritten, immediately
before the first VM session, and re-verified against the current VM
state (`hostname`, `uname -r`, `nvidia-smi`) per the brief's resume
procedure. `/mnt` on the VM is an ephemeral resource disk — assume it's
wiped on every restart; nothing there is a source of truth.

---

## 14. Resume checkpoint (updated on every interruption)

**Last updated:** 2026-08-20 (Phase 3 completion — all local-only work
done up to the VM gate).

- **Current phase:** Phase 4 (Zero-transfer to SIGN) — **blocked on the
  Azure VM being started** for the M2–M5 legs; M1/M6 legs not yet run
  either (deliberately batched with the VM session per the "batch VM
  work" instruction — M1/M6 are quick enough to just run in the same
  sitting once the VM session opens, no need to split them out earlier).
- **Completed:** Phase 0 (audit + plan), Phase 1 (foundation — loaders,
  family grouping, sampling, family-aware metrics, 37 tests), Phase 2
  (dataset characterization — findings in §6), Phase 3 (dataset-origin
  classifier — findings inline in the Phase 3 entry above: 0.9555 /
  0.9235 macro F1 raw/normalized, strong domain-shift evidence).
- **Incomplete:** Phases 4–11.
- **Last safely persisted artifacts:** everything from Phase 1, plus
  `src/sign/characterization/{stats,embeddings,run_characterization,nltk_setup}.py`,
  `results/sign/characterization/corpus_stats.json` + `embeddings_2d.csv`
  + 5 PNGs, `src/sign/origin_classification/run_origin_classifier.py`,
  `results/sign/EXP-SIGN-001/` and `results/sign/EXP-SIGN-002/`
  (config/metrics/predictions each). **None of this has been committed
  to git yet** — all present only in the local working tree as of this
  checkpoint (commit only on explicit request).
- **VM status:** not started. **Phase 4 needs it next** — see the
  checkpoint report for the explicit ask and time estimate.
- **Next action when work resumes:** if the VM is running, proceed with
  Phase 4 (restore `scripts/sync_to_vm.sh`/`sync_from_vm.sh`/
  `verify_gpu.py`/`verify_kernel.sh` from git history at commit
  `7a17e8e~1`, verify the VM per the brief's resume procedure, then run
  M1/M6 zero-transfer locally-or-on-VM first since they're cheap, then
  M2→M3→M4→M5 in sequence, persisting each method's `results/sign/EXP-SIGN-0##/`
  immediately after it finishes). If the VM is not running yet, ask the
  user to start it and wait — do not proceed to Phase 4 without it, and
  do not begin Phase 7+ (SIGN Train use) under any circumstance until
  Phase 4's results are fully persisted here.
