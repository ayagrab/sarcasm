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

**UPDATE (2026-08-20, later same day): verified against the official
upstream source, investigation closed.** Fetched
`train.csv`/`dev.csv`/`test.csv` directly from the SIGN paper's own
GitHub release (Peled & Reichart, ACL 2017,
[github.com/Lotemp/SarcasmSIGN](https://github.com/Lotemp/SarcasmSIGN),
`corpus/` folder) and diffed byte-for-byte against this repo's
`data/raw/original_{train,dev,test}_dataset.csv`:

| Split | Official file | Local file | MD5 match? |
|---|---|---|---|
| Train | `corpus/train.csv`, 12,000 lines | `original_train_dataset.csv` | ✅ identical (`b232d49c...`) |
| Dev | `corpus/dev.csv`, 1,500 lines | `original_dev_dataset.csv` | ✅ identical (`dd45cdc1...`) |
| Test | `corpus/test.csv`, **1,470 lines** | `original_test_dataset.csv` | ✅ identical (`2c17f440...`) |

**All three are byte-identical to the official release — this repo's
copy is exactly correct, not corrupted, truncated, or edited.** Critically,
**the official `test.csv` file itself has only 1,470 lines**, not the
1,500 the paper's summary text (300 tweets × 5) describes — the 30-row
shortfall is a property of the dataset as actually published, present in
the source repository, not an artifact of anything in this project. The
paper's README describes the *design* (300 tweets/split, 5
interpretations each); the released Test file doesn't fully match that
description. **No further action needed**: not a bug to fix, nothing to
"restore" — this project's copy is the correct, complete, unmodified
official file, and every zero-transfer/family-structure result in this
document already accounts for the true 1,470/265 count rather than
assuming the paper's nominal 1,500/300.

**Original findings below, now understood in light of the above (kept for
the record, not still "open questions"):**

- **Test has 1,470 pairs, not the official 1,500** (30 short) — confirmed
  above to be inherent to the officially published file itself.
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
  are still correct).
- Every sarcastic original in SIGN is, by construction, `sarcastic`; every
  interpretation is `not_sarcastic`. SIGN is **never** to be described as
  "15,000 independent sarcastic examples" anywhere in this phase's
  documentation — it is 3,000 originals + 15,000 interpretations
  (nominally; 14,970 actually present, see above), organized in families.

### Interpretation rank and the "primary reference" clarification (added 2026-08-20)

**Decision:** interpretation #1 (the first interpretation for a given
family, in the exact row order already present in the raw source file)
is treated as that family's **primary/best human reference**, not
interchangeable with #2–#5. This changes how "1 interpretation per
original" is selected everywhere in this document: **always
interpretation #1 specifically, never a random draw.**

- The family-aware loader (`src/sign/data/load_sign.py`) already
  preserves this order as `interp_index` (1..N, in file order) — no
  schema change was needed, just the explicit semantic designation. A
  convenience column `is_primary_interpretation` (`interp_index == 1`)
  was added for clarity.
- **Honest caveat, not a silent assumption**: the raw files carry no
  interpretation-ID column (confirmed again by the official-source diff
  above — the published files are plain `[original, interpretation]`
  pairs, nothing more). "Interpretation #1" therefore means "first row
  for this family in the officially published file," which is the only
  ordering signal available — there's no independent way to confirm this
  matches whatever internal numbering the original annotation process
  used. Treated as the best available proxy, disclosed as a limitation.
- **`src/sign/data/family_utils.py` now has a rank-based, non-shuffled
  selector** (`select_top_k_interpretations_per_family` /
  `select_primary_interpretation_per_family`) that picks interpretations
  #1..k by rank, always nested (k=1 ⊂ k=2 ⊂ k=3 ⊂ k=5) by construction —
  no seed needed since there's nothing random. The earlier seeded-shuffle
  selector (`select_k_interpretations_per_family`) is kept in the module,
  documented as deprecated for the primary-reference experiments, not
  deleted (no code depended on it yet — Phase 7 hadn't run).
- **Affects Phase 5, 7, 9, 10 methodology** — see each phase's updated
  entry below. Phase 4's zero-transfer evaluation already covers *every*
  interpretation (not just #1), so it's unaffected by this decision;
  Phase 5's analysis of those same predictions is where the primary-vs-
  full-family distinction actually enters.

### Task A vs. Task B vs. Primary-Reference: the SIGN evaluation must always specify which question it's answering (added 2026-08-20)

**This is a hard reporting rule from here through Phase 11, applied
retroactively to how Phase 4's existing M1–M3 results are *interpreted*
(not re-run):** every SIGN original tweet is, by construction, sarcastic;
every one of its human interpretations is, by construction,
not_sarcastic. Evaluating "originals only" and evaluating "originals +
interpretations combined" are **different research questions with
different appropriate primary metrics**, and a result from one must never
be quoted as if it were the other.

- **Task A — Original SIGN sarcasm transfer.** Eval set = the 265 SIGN Test
  originals only. Every gold label is `sarcastic` — this is **not** a
  balanced binary classification setting, so **Macro F1 is not the
  primary metric here**. Primary metric: **sarcasm detection
  rate / sarcastic recall** (correct sarcastic predictions ÷ total
  originals), plus false-negative count and rate, plus per-example
  predictions. Answers: "can a model developed on Dataset A recognize an
  independent SIGN sarcastic tweet at all?"
- **Task B — Full contrastive/binary SIGN evaluation.** Eval set = all
  1,735 rows (265 originals + 1,470 interpretations, both roles, all
  ranks). Binary metrics **are** appropriate here: Macro F1, accuracy,
  per-class precision/recall, confusion matrix. Answers the harder
  question: "can the model tell a sarcastic original apart from a sincere
  human rewrite of the same underlying meaning?" **This is the task
  Phase 4's existing Macro F1 numbers (§7) already measure** — they were
  always Task B numbers; the correction is in how they get *described*,
  not in the numbers themselves.
- **Primary-Reference view — a third, focused evaluation**: original vs.
  interpretation #1 only (one sarcastic + one best sincere reference per
  family, naturally balanced 1:1). Reports the same binary metrics as
  Task B plus **pair success rate** (original predicted sarcastic **and**
  interpretation #1 predicted not_sarcastic, both correct, per family).
  Distinct from Task B because it isolates the model's best-case
  contrastive signal from the noisier full-family aggregate.
- **Per-interpretation-rank breakdown**: not_sarcastic recall computed
  separately for interpretation #1, #2, #3, #4, #5 (where each rank has
  data). Tests — does not assume — whether a model is especially good at
  recognizing the *primary* interpretation as sincere, or behaves
  similarly across all five.
- **Reporting rule:** any sentence of the form "model X scores Y% on
  SIGN" is incomplete and must not be written — always name the task
  (Task A / Task B / Primary-Reference) the number belongs to. In
  particular: **a low Task B Macro F1 is not evidence a model "can't
  detect SIGN sarcasm"** — check Task A's detection rate before drawing
  that conclusion; the two can and do diverge sharply (see §7's revised
  M2/M3 interpretation).
- **Applies to:** Phase 4's existing results (§7, reinterpreted below,
  numbers unchanged), Phase 5 (formalizes Task A/B/Primary-Reference as
  named, separately-reported metric blocks per method), Phase 6 (false
  negatives = Task A misses; false positives = Task B's sincere
  interpretations mislabeled sarcastic — a distinct failure mode
  investigated on its own terms, see Phase 6's updated entry), Phase 8's
  before/after-adaptation comparison, and Phase 11's final report
  structure (§ Phase 11 below).

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
  per tweet (vs. just interpretation #1) improve the classifier's learned
  boundary?
- **RQ5** (added 2026-08-20): what kinds of SIGN sarcasm fail to
  transfer across datasets, and which of those failures does domain
  adaptation actually fix?

Phases 1–3 below build the evidence base (is there even a domain shift to
explain?); Phases 4–6 answer RQ2 and RQ5 and characterize *why*; Phases
7–10 answer RQ3/RQ4; Phase 6 is repeated after adaptation (§ Phase 8-10)
to close RQ5; Phase 11 synthesizes all of it.

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
  artifact. A monitor watches the VM log and rsyncs each method's
  `results/sign/EXP-SIGN-01#/` back to the local Mac **the moment that
  method's line is printed** (not waiting for the whole batch) — added
  2026-08-20 per explicit request, since `/mnt` is ephemeral and a VM
  loss should cost at most the one method currently mid-run.
- **Status: COMPLETE as of 2026-08-20.** VM session opened, environment
  rebuilt and verified (§13 log). **Verified state (not inferred from
  what was scheduled):**
  - M1 (EXP-SIGN-011): **DONE**, persisted locally, results in §7.
  - M6 (EXP-SIGN-016): **DONE**, persisted locally, results in §7.
  - M2 (EXP-SIGN-012): **DONE**, persisted locally, results in §7.
  - M3 (EXP-SIGN-013): **DONE**, persisted locally (briefly missing from
    the local Mac despite being computed on the VM — caught and re-synced
    at the Phase 4 checkpoint), results in §7.
  - M4 (EXP-SIGN-014): **DONE**, persisted locally, results in §7.
  - M5 (EXP-SIGN-015): **DONE**, persisted locally, results in §7.
  - **All six methods' `results/sign/EXP-SIGN-0{11..16}/` verified present
    locally (metrics.json + predictions.csv + config.json each) and
    git-committed.**
  - **Depends on:** Phase 1 (done). **Must complete and be persisted
    before Phase 7** — done; per the checkpoint gate above, Phase 5 and
    Phase 6 still come before Phase 7's first SIGN Train touch.

**Checkpoint gate (explicit, 2026-08-20 request): after Phase 4 finishes
and before any Phase 7 SIGN Train exposure**, stop and do three things in
order: (1) back up all six methods' zero-transfer `predictions.csv` +
`metrics.json` (already git-committed as each method lands, per the
per-phase-commit policy below — this step just confirms nothing from
Phase 4 is only on the ephemeral `/mnt` disk); (2) produce one
consolidated M1–M6 comparison table (macro F1, sarcasm detection rate,
false negative rate side by side — folded into Phase 5's output below
rather than a separate artifact); (3) only then proceed to Phase 6's full
false-negative analysis, and only after Phase 6 does Phase 7 (first SIGN
Train touch) begin. This does not change phase order — Phase 5 → 6 → 7
was already the plan — it makes the backup + single comparison table an
explicit, checked step rather than implicit in Phase 5/7's existing
inputs.

### Phase 5 — SIGN contrastive / family-aware evaluation

- **Research question:** RQ2, at family granularity — not just "is each
  row right" but "does the model correctly separate an original from its
  own rewrites."
- **Inputs:** Phase 4's predictions (no new model inference — this phase
  re-analyzes Phase 4's `predictions.csv` files with family-aware
  metrics, via `src.sign.family_eval.metrics`, already built and tested
  in Phase 1). **No SIGN Train use.**
- **Reports Task A, Task B, and Primary-Reference as three explicitly
  labeled, never-conflated blocks per method (§1's naming — note these
  are distinct from this phase's own "View 1/View 2" family-granularity
  split below; both distinctions coexist):**
  - **Task A** (originals only): sarcasm detection rate, FN count/rate —
    already computed in Phase 4 (`sign_originals_summary` in each
    method's `metrics.json`), just surfaced here alongside the rest
    rather than recomputed.
  - **Task B** (full 1,735-row set): Macro F1, accuracy, per-class
    precision/recall, confusion matrix — already computed in Phase 4,
    surfaced here.
  - **Primary-Reference** (original vs. interpretation #1 only, new
    computation this phase): accuracy, Macro F1, sarcastic
    precision/recall, not_sarcastic precision/recall, **pair success
    rate** (both original→sarcastic and interp#1→not_sarcastic correct,
    per family).
  - **Per-interpretation-rank breakdown** (new computation this phase):
    not_sarcastic recall computed separately for interp #1, #2, #3, #4,
    #5 — measured, not assumed, whether rank #1 is actually easier for
    the model than #2–#5.
- **Two family-granularity views on top of the above, both reported,
  never conflated (added 2026-08-20 per the interpretation-#1-is-primary
  decision, §1) — these answer "does the model separate an original from
  its own rewrites," a step beyond Task A/B/Primary-Reference's per-row
  metrics:**
  - **View 1 (primary-reference family view)** — original vs.
    interpretation #1 only (`is_primary_interpretation`). Same pairing as
    the Primary-Reference block above, reported here as
    pairwise/strict/soft family metrics instead of row-level binary ones.
  - **View 2 (full-family view)** — original vs. all available
    interpretations (1–5, whatever that family actually has). Original
    detection rate, interpretation non-sarcasm rate, pairwise contrastive
    accuracy, strict family accuracy (**all** available interpretations
    must be correct, not just #1), soft family score. Computed once over
    **all families** and reported again over the **clean (exactly-5)**
    subset only, so a reader can see whether anomalous/incomplete
    families are skewing the aggregate.
- **Outputs:** `results/sign/family_eval/` — per method: Task A block,
  Task B block, Primary-Reference block, per-rank breakdown, View 1, and
  View 2×2 family-subsets, all as JSON/CSV; full per-family results table
  (family_id, original prediction, every interpretation's prediction,
  which view(s) it passed/failed) for Phase 6's error analysis to consume
  directly rather than re-deriving. Also: one consolidated
  `results/sign/family_eval/m1_m6_comparison.csv` — one row per method
  (M1–M6), with Task A / Task B / Primary-Reference / family-view
  headline numbers side by side, organized under clearly labeled column
  groups so the distinction is visible at a glance, not just in prose
  (added 2026-08-20 per the pre-Phase-7 checkpoint request above).
- **Compute:** trivial (re-aggregating existing predictions).
- **VM required:** NO.
- **Estimated time:** ~2h.
- **Status:** **COMPLETE (2026-08-20)**, actual time ~30min (well under
  estimate — the metric functions were already built and tested in Phase
  1, this phase was mostly assembly). `src/sign/family_eval/run_family_eval.py`,
  8 new tests (`tests/test_sign_family_eval_run.py`), 202/202 project
  tests passing. Results: §7's Phase 5 subsection above; artifacts
  `results/sign/family_eval/`. **Depends on:** Phase 4 (done).

### Phase 6 — Error analysis on SIGN (MANDATORY: complete, not representative-sample)

- **Research question:** RQ2/RQ5's "why" — what kinds of sarcasm fail to
  transfer, and what are false-positive interpretations reacting to.
- **Scope requirement (2026-08-20, explicit, supersedes any earlier
  "representative examples" framing):** this must cover **every** SIGN
  false negative for every zero-transfer method, not a curated sample.
  Representative examples belong in the write-up/report; the underlying
  analysis artifact must be exhaustive.
- **Inputs:** Phase 4/5 outputs (predictions + family metrics — both the
  primary-reference and full-family views).
- **Outputs:**
  - `results/sign/error_analysis/false_negatives.csv` (or `.jsonl`) — one
    row per **original that at least one model missed**, columns: `family_id`,
    `original_text`, per-model predicted label (one column per method,
    e.g. `pred_M1`, `pred_M2`, ... `pred_M6`), `n_models_missed`,
    `which_models_missed` (list), `interpretation_1`..`interpretation_5`
    (blank where unavailable), qualitative error tag(s), free-text notes.
    This is the exhaustive machine-readable artifact the brief requires —
    every false negative, not a sample.
  - `results/sign/error_analysis/false_positives.csv` — the mirror for
    interpretations flagged sarcastic: `family_id`, `interp_rank`,
    `interpretation_text`, per-model predicted label, `original_text`
    (for context), qualitative tag(s). **This is Task B's dominant
    failure mode for the LLM methods (§7 — M2 76.5%, M3 still high FP
    rate on interpretations) and gets equal analytical weight to the
    false-negative side, not a footnote.** Test (don't assume) hypotheses
    for *why* a sincere rewrite still reads as sarcastic to the model:
    semantic/topic inheritance from the sarcastic original, negative
    sentiment mistaken for sarcasm, retained lexical cues, general
    over-prediction of the sarcastic class, interpretation quality/fidelity
    to the original's meaning, and any other structural property the data
    surfaces. **Central framing for this half of the analysis (added
    2026-08-20, explicit research issue, not just an error category):**
    SIGN's original/interpretation pairs are semantically related but
    differ in *form* (sarcastic vs. sincere expression of the same
    underlying meaning) — if a model tags both as sarcastic, check
    whether it's actually responding to topic/sentiment/underlying
    negative meaning rather than sarcastic linguistic form specifically.
    This is a candidate top finding for Phase 11, not a routine tag.
  - **Quantitative comparison, not anecdotal** (explicit requirement): for
    every proposed error category/characteristic (length, sentiment,
    punctuation, capitalization, lexical features, embedding position),
    compute it for **both** missed and correctly-detected originals and
    report the contrast — a characteristic common among misses that's
    equally common among successes is not reported as explanatory.
  - **Cross-model overlap analysis**: which originals are missed by every
    method, by most, by exactly one; per-model-pair miss overlap (a
    model×model or model×example matrix); whether DSPy's errors differ in
    *kind* from plain-prompted Qwen's, not just in count; category-level
    miss rates per method where the qualitative tagging supports it.
    Plots as useful (heatmap / overlap matrix), not obligatory for every
    cut.
  - Qualitative write-up (representative examples + conclusions) appended
    to this document and `EXPERIMENT_LOG.md`, explicitly titled to match
    the final-report requirement ("What Kinds of SIGN Sarcasm Do the
    Models Fail to Detect?" — drafted here, finalized in Phase 11).
- **Repeated after adaptation** (new, §Phase 8-10): the same artifacts are
  regenerated for the adapted model(s) once Phase 8/9 produce a selected
  adapted checkpoint, plus an explicit before/after diff (fixed / still
  missed / newly broken) — see Phase 8's entry.
- **Compute:** trivial to generate; the quantitative contrasts and
  cross-model overlap are systematic (not manual), the qualitative
  tagging/write-up is the part that takes real time.
- **VM required:** NO.
- **Estimated time:** ~4–6h (revised up from the original 2–4h given the
  "complete, not sampled" + cross-model overlap + quantitative-contrast
  requirements — qualitative tagging of up to ~265 originals × up to 6
  methods, plus building the overlap/contrast artifacts, is real work).
- **Status:** NOT STARTED. **Depends on:** Phase 5.

### Phase 7 — Prepare SIGN Train variants (data prep only, no training yet)

- **Research question:** none directly — controlled data prep for Phase
  8/9/10, gated on Phase 4 being safely persisted first (brief's hard
  requirement).
- **Inputs:** SIGN **Train** — 2,292 families total (verified 2026-08-20
  against the official upstream file, §1), 2,185 "clean" (exactly-5)
  after the Phase-1 filter. Official spec is 2,400; the local/official
  file itself has 2,292 by exact-text grouping (§1 — confirmed inherent
  to the published data, not a local error).
- **Primary balanced condition (default, used unless a specific
  experiment says otherwise): original + interpretation #1 only**, via
  `select_primary_interpretation_per_family` — deterministic, rank-based,
  **never a random draw** (2026-08-20 decision, §1). For the 2,292
  available Train families this gives **2,292 sarcastic + 2,292
  non-sarcastic** examples (naturally 1:1 balanced by construction, no
  imbalance-handling needed for this condition specifically).
- **Multi-interpretation conditions (k=2/3/5) — secondary/ablation, not
  the default**: `select_top_k_interpretations_per_family(df, k)`,
  always interpretations **#1..k by rank** (nested: k=1 ⊂ k=2 ⊂ k=3 ⊂
  k=5), never shuffled. These are inherently imbalanced (k non-sarcastic
  per 1 sarcastic) — **imbalance-handling policy, decided and documented
  here rather than silently at run time:**
  - **M1 (TF-IDF+LR):** `class_weight="balanced"` in
    `LogisticRegression` — trivial, already the exact pattern used for
    Phase 3's origin classifier (`src/sign/origin_classification/run_origin_classifier.py`).
  - **M6 (DeBERTa/HF `Trainer`):** rather than a custom weighted loss
    (a real change to `finetune.py`'s training loop, more invasive),
    **the sarcastic original is duplicated k times** so the training set
    stays exactly 1:1 for every k (k originals-by-duplication : k
    interpretations #1..k). Simpler, deterministic, reproducible, and
    isolates the question Phase 10 actually asks (does interpretation
    *diversity* help) from a confound of class imbalance or raw example
    count — which is the brief's explicit requirement (§4/§14 of the
    kickoff prompt). Documented per-experiment in each run's `config.json`
    (`imbalance_strategy: "duplicate_original_k_times"`).
- **Outputs:** `data/sign/train_variants/` — one CSV per condition
  (`primary.csv`, `k2.csv`, `k3.csv`, `k5.csv`, plus the Phase 9
  fractional samples), each recording its exact family_ids, interp_index
  values used, and (for k>1) the imbalance strategy applied.
  `assert_no_family_leakage` run and saved as a check for every variant,
  not just at generation time.
- **Compute:** trivial (sampling ~2,292 families).
- **VM required:** NO.
- **Estimated time:** ~1–2h.
- **Status:** NOT STARTED. **Depends on:** Phase 4 (persisted) + Phase 1.

### Phase 8 — Domain adaptation (M1 + M6 only, see §3)

- **Research question:** RQ2/RQ3 — does SIGN Train exposure help, and
  does it hurt Dataset A performance (catastrophic forgetting check)?
- **Conditions ×2 models:** (A) Dataset-A-only [= Phase 4's zero-transfer,
  reused, not rerun], (B) Dataset A + SIGN Train, (C) SIGN Train only —
  each evaluated on SIGN Test **and** re-evaluated on Dataset A's own held-
  out TEST to check for forgetting. **B/C use the primary balanced
  condition (original + interpretation #1, Phase 7) as the default SIGN
  Train data**, per the 2026-08-20 clarification — not a k>1 variant
  unless a specific follow-up experiment says so.
- **Mandatory post-adaptation error-analysis repeat (2026-08-20, new):**
  once a B/C model is selected, Phase 6's exhaustive false-negative/
  false-positive artifacts are regenerated for it, plus an explicit
  **before → after diff**: which zero-transfer false negatives are now
  fixed, which remain, whether any new false positives appeared, and
  whether the error *categories* shifted (not just the count). This is
  what actually answers RQ5's second half ("which failures does
  adaptation fix") — a macro-F1-improved number alone does not.
- **Compute:** M1 — seconds/run, all local. M6 — ~15–30 min/run (scales
  with Part II's 22-min/6,706-example baseline), VM required.
- **VM required:** YES for M6's (B)/(C) runs (2 new fine-tunes; (A) is
  already-frozen `EXP-009`, reused not rerun).
- **Estimated time:** M1 ~30 min total. M6 ~1–1.5h VM time for both runs
  + eval. Post-adaptation error-analysis repeat: ~1–2h on top (reuses
  Phase 6's machinery, adds the diff).
- **Status:** NOT STARTED. **Depends on:** Phase 7, and Phase 4 fully
  persisted (hard gate).

### Phase 9 — Learning curve: how much SIGN Train is needed

- **Research question:** RQ3.
- **Inputs:** Phase 7's fractional family-level samples (0/10/25/50/75/100%
  of SIGN Train families), **using the primary balanced condition
  (original + interpretation #1) at every fraction** so the curve isn't
  confounded by interpretation count changing alongside data volume —
  that's Phase 10's question, kept strictly separate. Same SIGN Test
  throughout.
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
- **Inputs:** Phase 7's k=1/2/3/5 variants — **always interpretations
  #1..k by rank** (nested, never shuffled — 2026-08-20 decision, §1/§7),
  same full SIGN Train family set (100%, not swept — that's Phase 9) at
  every k. Class imbalance handled per Phase 7's documented policy
  (`class_weight="balanced"` for M1, original-duplicated-k-times for
  M6) so the comparison isolates interpretation *diversity* from raw
  example count or class-balance artifacts, per the brief's explicit
  requirement.
- **Outputs:** `results/sign/interp_count_ablation/` — same metric set as
  Phase 9, compared across the 4 interpretation-count conditions.
- **Compute:** M1 — trivial, local. M6 — 4 conditions × ~15–30 min ≈
  1–2h VM time.
- **VM required:** YES (M6 leg only).
- **Estimated time:** M1 ~45 min. M6 ~1–2h VM time.
- **Status:** NOT STARTED. **Depends on:** Phase 7, Phase 4 persisted.

### Phase 11 — Final synthesis

- **Research question:** RQ1–RQ5, connected end-to-end.
- **Outputs:** a new "Part III — SIGN Generalization" section in
  `PROJECT_SUMMARY.md` (clean narrative + final tables), a completed
  `EXPERIMENT_LOG.md` SIGN section (full audit trail, every
  `EXP-SIGN-###`), this document marked fully COMPLETED. **Required
  section structure (2026-08-20, explicit — the Task A/B/Primary-
  Reference/family distinctions from §1 must stay visible as separate
  named sections, never collapsed into one narrative):**
  1. Cross-Dataset Sarcasm Detection (Task A — SIGN originals only)
  2. Contrastive Sarcasm Recognition (Task B — original vs. sincere
     interpretations)
  3. Primary Human Reference Evaluation (original vs. interpretation #1)
  4. Full Family Evaluation (original + all five interpretations)
  5. **What Kinds of SIGN Sarcasm Do the Models Fail to Detect?** (RQ5 —
     quantitative + qualitative, drafted in Phase 6, finalized here)
  6. Why Are Sincere SIGN Interpretations Misclassified as Sarcastic?
     (Task B's false-positive analysis, Phase 6, incl. the
     semantic-meaning-vs-sarcastic-form finding if it holds up)
  7. Effect of SIGN Domain Adaptation (Phase 8, before/after diff)
  8. Effect of the Amount of SIGN Training Data (Phase 9, learning curve)
  9. Effect of the Number of Human Interpretations (Phase 10, k-ablation)
  10. Remaining Difficult Cases After Adaptation (Phase 8's post-adaptation
      error-analysis repeat)
- **VM required:** NO.
- **Estimated time:** ~2–4h writing/assembly, after all prior phases.
- **Status:** NOT STARTED. **Depends on:** everything above.

---

## 5. Time estimates — summary

| Phase | Local/dev time | VM time | VM required | Status |
|---|---:|---:|:---:|---|
| 0 — Audit & plan | done | — | NO | ✅ DONE |
| 1 — Foundation | done (~1h actual) | — | NO | ✅ DONE |
| 2 — Characterization | done (~25min actual) | — | NO | ✅ DONE |
| 3 — Origin classifier | done (~25min actual) | — | NO | ✅ DONE |
| 4 — Zero-transfer | M1/M6 done locally | **~2.9h revised** (M2–M5, see below) | YES | 🔄 IN PROGRESS |
| 5 — Family eval | 2–3h (primary + full-family views) | — | NO | NOT STARTED |
| 6 — Error analysis | **4–6h** (revised up: complete not sampled) | — | NO | NOT STARTED |
| 7 — SIGN Train prep | 1–2h | — | NO | NOT STARTED |
| 8 — Domain adaptation | 0.5h (M1) + ~1–2h (post-adapt error repeat) | 1–1.5h (M6) | YES | NOT STARTED |
| 9 — Learning curve | 1h (M1) | 2–3.5h (M6) | YES | NOT STARTED |
| 10 — Interp-count ablation | 0.75h (M1) | 1–2h (M6) | YES | NOT STARTED |
| 11 — Synthesis | 2–4h | — | NO | NOT STARTED |
| **Total remaining from now** | **~15–21h local/dev** | **~5–7h VM** | — | |

**Phase 4 VM time, revised with a real measured rate** (2026-08-20, mid-run):
M2 zero-shot measured at **~1.55s/example** (1,533/1,735 done in 32m29s) →
M2 total ≈ **44–45 min** (very close to the original ~40–60min estimate).
Applying Part II's DEV-time *relative* pacing between methods (M3 ≈
2.3–2.6× M2's rate, M4 ≈ 1.7–1.9×, both per-example, scaled onto SIGN's
1,735-row eval set) gives revised estimates: **M3 ≈ 1h45m–2h05m, M4 ≈
1h15m–1h25m, M5 ≈ 45min–1h15m** (inference-only against the frozen
program, expected closer to M2/M4's pace than M3's, no compile cost).
**Revised Phase 4 VM total from the point M2 finishes: ≈ 3h45m–4h45m**;
already ~35–40 min into that as of this estimate (M2 in progress) →
**≈ 3h10m–4h10m remaining** for M3+M4+M5 combined. Will be updated again
once each method's *actual* wall-clock lands (M2's real number already
folded in above; M3/M4/M5 still projected).

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

*(Phase 4, in progress, started 2026-08-20. VM session opened this date
— see §13 for the setup log. This section is the phase's most important
scientific artifact; adaptation results get their own §8/§9/§10 sections,
never merged into this one.)*

**Read this table as two separate tasks, not one score per method** (see
§1's Task A/B/Primary-Reference clarification, added 2026-08-20): the
"Sarcasm detection rate" column is **Task A** (originals only, single
gold class — this is the answer to "can the model recognize a SIGN
sarcastic tweet"); Accuracy/Macro F1/confusion matrix are **Task B**
(full 1,735-row contrastive set — the answer to "can the model tell that
original apart from a sincere rewrite"). A method can score well on one
and poorly on the other — see M2/M3 below.

**Eval set for every method below:** SIGN Test, all roles, full set
(n=1,735: 265 originals + 1,470 interpretations — see §1 for why this is
1,735/300 rather than the official 1,800/300).

| Method | Experiment ID | Status | Task B Accuracy | Task B Macro F1 | Task A: sarcasm detection rate | Task A: FN rate | Confusion matrix `[[TN,FP],[FN,TP]]` (not_sarcastic, sarcastic) |
|---|---|---|---:|---:|---:|---:|---|
| M1 TF-IDF+LR | EXP-SIGN-011 | done | 0.3660 | 0.3563 | 0.7962 (211/265) | 0.2038 | `[[424,1046],[54,211]]` |
| M6 DeBERTa-v3-base | EXP-SIGN-016 | done | 0.5326 | 0.4724 | 0.6377 (169/265) | 0.3623 | `[[755,715],[96,169]]` |
| M2 Qwen zero-shot | EXP-SIGN-012 | **done** | 0.3418 | 0.3397 | **0.9358 (248/265)** | 0.0642 | `[[345,1125],[17,248]]` |
| M3 Qwen few-shot | EXP-SIGN-013 | **done** | 0.4133 | 0.4015 | 0.8943 (237/265) | 0.1057 | `[[480,990],[28,237]]` |
| M4 Qwen reasoning | EXP-SIGN-014 | **done** | 0.3782 | 0.3658 | 0.9132 (242/265) | 0.0868 | see `results/sign/EXP-SIGN-014/metrics.json` |
| M5 DSPy MIPROv2 (frozen, inference-only) | EXP-SIGN-015 | **done** | 0.4092 | 0.3957 | 0.7736 (205/265) | 0.2264 | see `results/sign/EXP-SIGN-015/metrics.json` |

**Phase 4 status: COMPLETE as of 2026-08-20 — all 6 methods (M1, M2, M3,
M4, M5, M6) zero-transfer evaluated, predictions + metrics persisted
locally and committed.** M3's results (`EXP-SIGN-013`) were briefly
missing from the local Mac despite being computed on the VM — caught at
this checkpoint and re-synced before proceeding; all six now verified
present.

**M1/M6 interpretation — a striking reversal of Part II's headline
finding.** On Dataset A, M1 and M6 (the two "trained on labels" methods)
were the *balanced* ones (Part II: M1 FP=157/FN=174, M6 FP=112/FN=122),
while every LLM-prompted method was heavily FP-skewed. **On SIGN,
zero-transfer, both M1 and M6 flip to heavy false-positive bias
themselves**: M1 misclassifies 1,046/1,470 (71.2%) of SIGN's sincere
interpretations as sarcastic; M6 misclassifies 715/1,470 (48.6%). Both
scores collapse well below their Dataset A TEST numbers (M1: 0.7403 →
0.3563; M6: 0.8209 → 0.4724) — consistent with Phase 2/3's domain-shift
evidence, and a direct, quantified answer to "does the shift actually
hurt." M6 remains the stronger of the two on SIGN (higher macro F1,
lower FP rate) but both are far below their in-domain performance, and
both **still detect a majority of the sarcastic originals** (M1: 79.6%,
M6: 63.8%) — meaning the failure mode isn't "can't recognize the
sarcastic tweets at all," it's "can't tell sincere SIGN interpretations
apart from sarcastic ones" — exactly what Phase 2's embedding finding
(SIGN originals/interpretations overlap heavily with each other) predicted.
Full detail: `results/sign/EXP-SIGN-011/`, `results/sign/EXP-SIGN-016/`.

**M2 (Qwen zero-shot) interpretation — corrected 2026-08-20 to separate
Task A from Task B (numbers unchanged, framing fixed):**

- **Task A (SIGN originals, sarcasm detection):** M2 is the **best of the
  three methods so far** — 248/265 = **93.6% sarcastic recall**, FN rate
  6.4%. It is *not* correct to describe M2 as "~34% sarcasm-detection
  performance" — that 0.3397 figure is Task B's Macro F1, a different
  question. On Task A alone, M2 clearly can recognize unseen SIGN
  sarcastic tweets.
- **Task B (full contrastive, 1,735 rows):** M2's failure mode is
  entirely on the *interpretation* side, not the original side: **1,125/1,470
  (76.5%)** of sincere interpretations are misclassified as sarcastic —
  worse than either M1 (71.2%) or M6 (48.6%) — at a sarcastic precision
  of just 18.1% (`results/sign/EXP-SIGN-012/metrics.json`). Task B Macro
  F1 = 0.3397, barely above M1's 0.3563 and well below M6's 0.4724,
  *despite* M2's much higher Task A recall — Macro F1 is the right
  primary metric **for Task B specifically** (a single-class recall
  number would be misleading there), while Task A correctly uses
  detection rate instead. **Net read: M2 is not failing to detect
  sarcasm — it is failing to withhold the sarcastic label from sincere
  rewrites of the same underlying meaning** (see §1's "semantic meaning
  vs. sarcastic form" framing, expanded in Phase 6).

Full detail: `results/sign/EXP-SIGN-012/`.

**M3 (Qwen few-shot) interpretation — done 2026-08-20:**

- **Task A:** 237/265 = **89.4% sarcastic recall**, FN rate 10.6% —
  *lower* than M2's 93.6%, a real (if modest) trade-off, not noise (28
  vs. 17 missed originals out of 265).
- **Task B:** Macro F1 improves to **0.4015** (vs. M2's 0.3397), driven by
  better interpretation rejection: not_sarcastic recall on interpretations
  rises to 32.7% (480/1,470) from M2's 23.5% (345/1,470); sarcastic
  precision ticks up to 0.193 from 0.181.
- **Reading the trade-off explicitly (per §1's rule against conflating
  tasks):** few-shot prompting makes M3 *less* trigger-happy about
  labeling things sarcastic overall — this helps Task B (fewer false
  positives on interpretations) but costs a small amount of Task A recall
  (a few originals that zero-shot's more aggressive sarcastic-leaning
  bias happened to catch are now missed). Neither number "wins" outright;
  which matters more depends on whether the downstream use case cares
  more about not missing sarcasm (Task A) or not over-flagging sincere
  text (Task B) — both are real, both are reported, deliberately not
  collapsed into one verdict here.

Full detail: `results/sign/EXP-SIGN-013/`.

**Pending, per §1's rule:** primary-reference (original vs. interp #1
only) and per-rank (#1–#5) not_sarcastic-recall breakdowns are not yet
computed for any method — that's Phase 5's job, run once against all of
M1–M6's persisted predictions rather than piecemeal per method here.

### Phase 5 results — family-aware / contrastive evaluation (2026-08-20, COMPLETE)

Computed by `src.sign.family_eval.run_family_eval` over all 6 methods'
persisted `predictions.csv`, no new inference. Outputs:
`results/sign/family_eval/<EXP-ID>/metrics.json` (full per-method detail)
and `results/sign/family_eval/m1_m6_comparison.csv` (the consolidated
table). Consistency check passed: recomputed Task A/B numbers match §7's
Phase-4 numbers exactly (e.g. M1 Task A 0.7962, M2 Task B Macro F1
0.3397) — confirms no drift between the two computations.

| Method | Primary-Ref Macro F1 | Primary-Ref pair success | View1 strict family acc. | View2(all) strict family acc. | View2(all) soft family score |
|---|---:|---:|---:|---:|---:|
| M1 TF-IDF+LR | 0.5172 | 13.6% | 13.6% | 0.0% | 0.178 |
| M2 Qwen zero-shot | 0.5408 | 20.8% | 20.8% | 1.1% | 0.212 |
| M3 Qwen few-shot | 0.5901 | 26.8% | 26.8% | 1.9% | 0.282 |
| M4 Qwen reasoning | 0.5645 | 23.0% | 23.0% | 0.8% | 0.236 |
| M5 DSPy frozen | 0.5650 | 24.2% | 24.2% | 2.3% | 0.218 |
| M6 DeBERTa | 0.5658 | 21.5% | 21.5% | 3.8% | 0.370 |

**Finding 1 — the Primary-Reference view confirms interpretation #1 is
genuinely an easier contrastive case, for every method.** Primary-Ref
Macro F1 (0.52–0.59) is well above the corresponding full Task B Macro F1
(0.34–0.47, §7) across the board — e.g. M1 jumps from 0.356 to 0.517, M6
from 0.472 to 0.566. This validates treating interpretation #1 as the
"clean, minimal-noise" reference the primary-reference decision (§1)
assumed it would be — but pair success rate (both original and
interpretation #1 correct, same family) is still only 13.6–26.8%, so even
the easiest single-pair case is hard in absolute terms.

**Finding 2 — strict family accuracy (all 5 interpretations + original
correct) is near zero for every method (0–3.8%), while soft family score
(mean fraction of a correctly-detected family's interpretations also
correct) sits around 0.18–0.37.** The gap between View 1's per-pair
success and View 2's all-or-nothing strict accuracy is the clearest
illustration yet of Task B's difficulty: getting *one* interpretation
right per family is plausible, getting *all five* simultaneously right is
not, for any method tested.

**Finding 3 — per-interpretation-rank recall does *not* show a clean
"rank #1 is easiest" pattern — tested, not assumed, per §1's explicit
instruction not to assume an ordering effect.** Full per-rank table in
`m1_m6_comparison.csv`; e.g. M6's not_sarcastic recall by rank is
49.8% / 54.7% / 39.6% / 60.4% / 53.2% (#1→#5) — non-monotonic, rank #4
highest, rank #3 lowest, not a decaying-with-rank curve. M1 shows a
mild, mostly-monotonic decline (29.8% → 25.3%) while M3/M4/M5 all peak
at rank #4, not #1. **Conclusion: interpretation #1 being the "primary
reference" (§1) is a provenance/quality decision (first-listed = best
annotation), not evidence that it is linguistically the easiest
interpretation for a model to recognize as sincere** — those are
different claims, and only the first is actually established by this
data.

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
- [x] Phase 4 — Zero-transfer to SIGN (M1–M6) — **COMPLETE, all 6 methods
      persisted and committed**
- [x] Phase 5 — SIGN contrastive/family evaluation — **COMPLETE**
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

### VM session log (2026-08-20, current session)

- **Verified against the known-good baseline** (`scripts/verify_kernel.sh`,
  extracted from git history and run inline via SSH, not re-added to the
  tracked repo): hostname `dpmlgpuNC6sv32025s-0003`, kernel
  `6.8.0-1029-azure` (match), `nvidia-smi` healthy, both Tesla M60s free.
- **`/mnt` was wiped again** (as expected/documented) — rebuilt
  `/mnt/vmadmin/{projects/sarcasm,sarcasm-env,huggingface}` (came back
  root-owned, needed `sudo mkdir`/`sudo chown`, per the established
  pattern), synced the repo (incl. the new `src/sign/`) via `rsync`
  (mirroring `sync_to_vm.sh`'s exclude list), rebuilt the Python venv
  from `environment_stage_b.txt`'s pinned `pip freeze` with two
  necessary deviations from the frozen list (both consistent with the
  project's established "bump the closest available version, don't
  debug it" policy for this exact class of problem):
  - `litellm==1.96.1` → `1.96.2` (1.96.1 removed from PyPI, same fix as
    the 2026-08-16 incident recorded in `EXPERIMENT_LOG.md`).
  - `torchaudio==2.5.1+cu118` / `torchvision==0.20.1+cu118` **removed
    entirely** (new deviation, not previously encountered) — neither
    package's `+cu118` build exists on plain PyPI (only on
    `download.pytorch.org`'s dedicated index, which only `torch` itself
    was explicitly pointed at), and neither is imported anywhere in this
    codebase (`grep -rn "torchaudio\|torchvision" src/ scripts/ config/`
    → no hits) — confirmed unused before removing, not assumed.
  - Torch itself: `pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118`
    (exact pinned version, correct CUDA build) — installed cleanly,
    `torch.cuda.is_available() == True` confirmed on both GPUs.
- Test suite on VM: 98/98 passing for the classification + SIGN scope
  (`tests/test_sign_*.py` + `tests/test_classification*.py`); the
  remaining 9 pre-existing failures are Part I-only tests
  (nltk/google-generativeai/seaborn) that were never in the Stage B
  environment's scope, not a regression.
- `Qwen/Qwen3-4B-Instruct-2507` re-downloaded fresh into
  `/mnt/vmadmin/huggingface` (21.6s, 7.6GB — matches the "~9.3GB combined
  Qwen+DeBERTa, under a minute" figure from prior recoveries).
  `results/EXP-008/compiled_program.json` (the frozen MIPROv2 program,
  1.3KB) copied over individually via `scp` for M5's inference-only run
  (not part of the `rsync` exclude-`results/` sync).
- Smoke-tested end-to-end (`--limit 5` on M2 zero-shot) before launching
  the real run — passed, ~1.7s/example, matching Part II's rate.

---

## 14. Resume checkpoint (updated on every interruption)

**Last updated:** 2026-08-20, Phase 4 checkpoint (VM still up, idle — no
job queued as of this update). Every status below was **verified against
actual persisted artifacts** (`results/sign/EXP-SIGN-0##/metrics.json`
existing on the local Mac) — not inferred from what was scheduled to run.

### CURRENT STATUS

**Phase 4 (Zero-transfer to SIGN) COMPLETE. Phase 5 (family-aware/contrastive
evaluation) COMPLETE.** All 6 methods' Task A/B/Primary-Reference/per-rank/
family-view metrics computed and persisted (`results/sign/family_eval/`).
At the checkpoint gate (§1/§Phase 4-5): backup verified, comparison table
done → Phase 6 (mandatory complete error analysis) is next → only then
Phase 7 (first SIGN Train touch).

### LAST SAFE CHECKPOINT

All of Phase 4: `results/sign/EXP-SIGN-0{11,12,13,14,15,16}/` each
verified present locally (config.json + metrics.json + predictions.csv)
and committed to git (commits `a26b990`, `d3184a8`, `3e1e19c`, plus this
checkpoint's commit for M3/M5). Nothing from Phase 4 depends on the
ephemeral VM `/mnt` disk anymore.

### CURRENT EXPERIMENT

None running. VM is idle (M2-M4 process and M5 process both exited
cleanly, verified via `pgrep`). No further VM work is needed until Phase
8 (M6 domain adaptation) — Phase 5, 6, and 7 (data prep) are all
local-only.

### COMPLETED EXPERIMENTS (this phase)

| Experiment | Method | Task B Macro F1 | Task A detection rate | Persisted locally? |
|---|---|---:|---:|---|
| EXP-SIGN-001 | Origin classifier, raw text | 0.9555 | n/a | ✅ |
| EXP-SIGN-002 | Origin classifier, normalized text | 0.9235 | n/a | ✅ |
| EXP-SIGN-011 | M1 zero-transfer | 0.3563 | 79.6% (211/265) | ✅ |
| EXP-SIGN-012 | M2 zero-transfer | 0.3397 | 93.6% (248/265) | ✅ |
| EXP-SIGN-013 | M3 zero-transfer | 0.4015 | 89.4% (237/265) | ✅ |
| EXP-SIGN-014 | M4 zero-transfer | 0.3658 | 91.3% (242/265) | ✅ |
| EXP-SIGN-015 | M5 zero-transfer | 0.3957 | 77.4% (205/265) | ✅ |
| EXP-SIGN-016 | M6 zero-transfer | 0.4724 | 63.8% (169/265) | ✅ |

Plus Phase 1 (loaders/tests) and Phase 2 (characterization) artifacts —
see their entries above.

### REMAINING EXPERIMENTS (this phase)

None — Phase 4 is complete.

### BLOCKERS

None currently. (Historical, resolved: `litellm==1.96.1` unavailable on
PyPI → bumped to 1.96.2; `torchaudio`/`torchvision` `+cu118` builds
unavailable off the PyTorch index → removed, confirmed unused; a
Monitor-based auto-sync-back had a shell pipe-buffering bug → replaced
with a polling-based sync-back; two background Monitor tasks were lost
across a `/compact` and had to be relaunched, and separately a zsh
word-splitting bug — `$VAR "cmd"` not splitting the way it does in bash —
broke their first relaunch attempt, fixed by writing the ssh invocation
literally instead of through a variable; **M3's results
(`EXP-SIGN-013`) were computed on the VM but not yet synced/committed
locally when Phase 4 was first thought complete** — caught at this
checkpoint by explicitly verifying all 6 `metrics.json` files present
before declaring Phase 4 done, re-synced, now resolved.)

### NEXT ACTION

Phase 6 (mandatory complete error analysis, local, no VM): build the
exhaustive false-negative/false-positive CSVs (every SIGN original missed
by at least one method, every interpretation flagged sarcastic), the
quantitative missed-vs-detected contrast, and the cross-model overlap
analysis, per the phase's updated entry above. No SIGN Train exposure.
VM can stay off for this and Phase 7 (next VM need is Phase 8's M6
domain adaptation) — ask the user before actually powering it down.

### Full artifact/environment state

- **Last safely persisted artifacts:** everything from Phase 1, plus
  `src/sign/characterization/{stats,embeddings,run_characterization,nltk_setup}.py`,
  `results/sign/characterization/corpus_stats.json` + `embeddings_2d.csv`
  + 5 PNGs, `src/sign/origin_classification/run_origin_classifier.py`,
  `results/sign/EXP-SIGN-001/`, `EXP-SIGN-002/`, `EXP-SIGN-011/`,
  `EXP-SIGN-016/`, `EXP-SIGN-012/` (config/metrics/predictions each),
  `src/sign/zero_transfer/{io,run_m1_zero_transfer,run_m6_zero_transfer,run_llm_zero_transfer,run_m5_zero_transfer}.py`,
  the interpretation-rank additions to `src/sign/data/{load_sign,family_utils}.py`
  and their tests. **None of this has been committed to git yet** — all
  present only in the local working tree (commit only on explicit
  request).
- **VM status:** running, actively executing M3. Two monitors active:
  one 15-min status-check heartbeat, one ~2-min polling sync-back (pulls
  any newly-`metrics.json`-complete `results/sign/EXP-SIGN-01#/` to the
  local Mac automatically).
- **Command/config/seed for the current experiment:** `python -m
  src.sign.zero_transfer.run_llm_zero_transfer` (all three modes,
  `--modes zero_shot few_shot reasoning` default), model
  `Qwen/Qwen3-4B-Instruct-2507`, `provider=local_hf`, `temperature=0.0`,
  `seed=42` (Dataset A TRAIN few-shot demo selection only — reproduces
  Part II's exact EXP-003 demo set, `select_random_few_shot`, n_shots=8).
  Eval data: `data/sign/family_table_test.csv` (1,735 rows, all roles).
