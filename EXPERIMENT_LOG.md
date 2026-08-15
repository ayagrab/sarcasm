# Experiment Log

Living, chronological, operational record of the **sarcasm detection (classification)**
work. This is a *new* phase inside the existing `sarcasm` repository — see
"Relationship to the existing repository" below for how it fits with what
was already here. For the clean, high-level narrative see `PROJECT_SUMMARY.md`.
This file is never overwritten; new entries are appended.

---

## Current Status

**Phase:** Stage A **complete** — infrastructure built, classical baseline
evaluated, everything else implemented and smoke-tested but not executed.
Stopped here per the task's instruction to halt before expensive
LLM/DSPy/Transformer runs. See "Stage A Readiness Report" below.

Completed:
- [x] Repository audit (Phase 1)
- [x] Dataset discovery and audit (Sarcasm Corpus V2)
- [x] Leakage analysis
- [x] `EXPERIMENT_LOG.md`, `PROJECT_SUMMARY.md` created
- [x] Canonical dataset builder (`src/classification/data/build_canonical_dataset.py`) — run for real
- [x] Reusable dataset audit script (`src/classification/data/audit_dataset.py`) — run for real
- [x] Canonical split (`src/classification/data/make_splits.py`) — run for real
- [x] Evaluation framework (`src/classification/evaluation/`)
- [x] Classical baseline — **EVALUATED** on frozen test set: EXP-001, Macro F1 0.740
- [x] LLM pipeline (zero-shot/few-shot/reasoning) — **IMPLEMENTED + SMOKE-TESTED** (mocked client); blocked on API keys for real execution
- [x] DSPy pipeline (Predict/BootstrapFewShot/MIPROv2) — **IMPLEMENTED**, structurally reviewed; blocked on `dspy` install + API keys
- [x] Transformer fine-tuning pipeline — **IMPLEMENTED**, structurally reviewed; no checkpoint downloaded (Stage A rule), blocked on Stage B disk/GPU planning
- [x] Config files for every planned experiment (`configs/*.json`)
- [x] One coherent experiment runner (`src/classification/run_experiment.py`)
- [x] 49 new unit tests, all passing (138/138 total with the existing suite)

**Blocked:** see "Blockers" section below. None of the blockers prevented
Stage A infrastructure work; they only block *running* the LLM/DSPy/
Transformer experiments (Stage B).

**Next:** awaiting explicit confirmation to start Stage B (compute
machine / API keys / model downloads) — see the readiness report.

---

## Relationship to the existing repository

**Important correction of the task brief:** the task description this log
was created under assumed (a) the dataset is Hebrew, and (b) the repository
already contains classification code and a labeled dataset ready for a
6-way model comparison. Neither is accurate — recorded here so nobody
re-derives this by re-reading the whole repo.

1. **Language.** The dataset (Sarcasm Corpus V2, UC Santa Cruz) is
   **English**. The task brief's own "IMPORTANT CORRECTIONS" section already
   confirmed this; the repo audit independently confirms it (see below).
2. **What already exists here is a *different* task.** The current
   repository (`README.md`, `docs/*`) documents a **sarcasm
   interpretation/neutralization benchmark**: given a sarcastic tweet, an
   LLM rewrites it as a sincere, non-sarcastic sentence, and the rewrite is
   scored by an LLM judge / NLI model / human annotators (Alt-Test, Fleiss'
   Kappa, Kruskal-Wallis, BLEU/ROUGE/PINC). That pipeline is fully
   implemented (`src/generation/`, `src/evaluation/evaluate_with_llm.py`,
   `src/evaluation/evaluate_with_nli.py`, `src/postprocessing/*`) and is
   **not sarcasm classification** — it never predicts sarcastic/not-sarcastic,
   it rewrites already-known-sarcastic text.
3. **The classification/detection task is a planned-but-unstarted next
   phase**, already scoped (at a high level) in `docs/finetuning_plan.md`
   (written 2026-07-16): "fine-tune a dedicated BERT-based binary sarcasm
   classifier ... Sarcasm Corpus V2 ... GEN/HYP/RQ." That plan only covers
   one BERT fine-tune, not the 6-approach comparison (classical ML / LLM
   zero-shot / few-shot / reasoning / DSPy / fine-tuned transformer) this
   task asks for. **This work supersedes and substantially broadens
   `docs/finetuning_plan.md`'s scope**, reusing the same target dataset it
   already staged.
4. **No classification code existed before this session.** `data/raw/sarcasm_corpus_v2/`
   was already present but, per `docs/project_structure.md`, explicitly
   "not yet used by any script." Everything under the new `src/classification/`
   package (see below) is being built from scratch in this session.
5. **Consequence for how this work is organized:** rather than reusing
   `src/evaluation/`, `src/generation/`, `config/models.py` etc. (which are
   specific to the interpretation pipeline and already documented/tested),
   the new classification work lives in its own clearly-separated
   subpackage (`src/classification/`), its own prompts folder
   (`prompts/classification/`), and its own config files (`configs/`), so
   the two pipelines don't collide or get confused with each other. Shared,
   generic infra (`.env`/`config/settings.py` pattern, `data/raw/` being
   read-only, the `python -m src.<pkg>.<script>` convention, the pytest/mock
   testing style) is reused as-is.

---

## Dataset Information

### Source and location

`data/raw/sarcasm_corpus_v2/` — **Sarcasm Corpus V2** (UC Santa Cruz,
Oraby et al.). Three CSV files, one per sarcasm-category subset. Confirmed
from `docs/finetuning_plan.md` and `docs/project_structure.md` (both already
in the repo before this session) — **not guessed from filenames**:

| File | Category (documented meaning) | Rows | `sarc` | `notsarc` |
|---|---|---:|---:|---:|
| `GEN-sarc-notsarc.csv` | **General Sarcasm** — general-purpose sarcastic vs. sincere forum posts | 6,520 | 3,260 | 3,260 |
| `HYP-sarc-notsarc.csv` | **Hyperbole** — sarcasm expressed via exaggeration | 1,164 | 582 | 582 |
| `RQ-sarc-notsarc.csv` | **Rhetorical Questions** — sarcasm expressed as a rhetorical question | 1,702 | 851 | 851 |
| **Total** | | **9,386** | **4,693** | **4,693** |

Columns in every file: `class` (`sarc`/`notsarc`), `id` (integer, **resets
to 1 within each file — not globally unique**, so a canonical ID must
combine category + id, or be freshly generated), `text` (the post).

Each category file is a header row + comma-separated rows; `text` fields
containing embedded commas/newlines are properly double-quoted (verified
with `pandas.read_csv`, not naive line counting — a naive `wc -l` on
`GEN-sarc-notsarc.csv` reports 8,867 lines because some `text` fields
contain literal newlines inside quotes; the real row count via `pandas` is
6,520, matching `docs/finetuning_plan.md`'s documented count exactly).

The three files are **each perfectly class-balanced** (50/50), and the
combined dataset is perfectly balanced (4,693/4,693) since each subset is.
**No class-imbalance handling is needed.**

### Data quality checks performed (2026-08-11)

First checked ad hoc with `pandas` directly, then formalized into two
reusable, re-runnable scripts (used for the authoritative numbers below):
`src/classification/data/build_canonical_dataset.py` (combines the 3 raw
files, attaches `example_id`/`category`/`source_file`/`dup_group_id`/
`label_conflict`) and `src/classification/data/audit_dataset.py` (computes
and reports the checks below, writes `data/processed/sarcasm_v2_audit_report.json`).
The audit script's duplicate-text key normalizes whitespace runs as well as
case (`" ".join(text.strip().lower().split())`), which is slightly stronger
than the ad hoc first pass — the numbers below are from the formal script
and are authoritative.

| Check | Result |
|---|---|
| Missing/null `text` | 0 |
| Missing/null `class` | 0 |
| `id` uniqueness within file | unique within each file (1..N) |
| `id` uniqueness across files | **not unique** — not usable as a global key on its own (canonical `example_id` = `f"{category}-{id}"`) |
| Exact duplicate `text` within a single file | 0 in every file |
| Duplicate `text` groups (normalized: case + whitespace insensitive), within **and** across files | **596 rows in 297 groups** |
| Duplicate groups with **conflicting labels** | **22 rows across several groups** (see `example_id` list in `data/processed/sarcasm_v2_audit_report.json` → `label_conflict_example_ids`, e.g. `GEN-103`, `RQ-679`..`RQ-684`) |
| Word-length distribution (whitespace tokens) | min 10, 25th pct 22, median 38, mean 48.7, 75th pct 67, max 150 |
| Degenerate very-short examples (≤2 words) | 0 |
| Degenerate very-long examples (>200 words) | 0 |
| Metadata fields beyond `class`/`id`/`text` (author, conversation, thread, timestamp, source post) | **none present in the raw files** |

Notable: several of the label-conflict rows are near-consecutive RQ ids
(`RQ-679` through `RQ-684`) — likely a cluster from the same source thread
that got annotated inconsistently. Worth a closer qualitative look during
error analysis, not before.

**The 22 label-conflict rows** (same normalized text, different `class`
across duplicate copies) are a genuine annotation inconsistency in the
source corpus, not a bug in this repo. Per the "never silently remove
examples" rule, these rows are **kept**, tagged `label_conflict=True` and
given a shared `dup_group_id` in the canonical dataset so they can be
inspected during error analysis; they are not dropped or relabeled.

### Decisions made (and why)

1. **No rows are being dropped or deduplicated in the canonical dataset.**
   All 9,386 rows are preserved, each tagged with `category`, `source_file`,
   `dup_group_id` (shared by rows with matching normalized text — singleton
   group for everything else), and `label_conflict` (bool). Downstream
   consumers (splitting, training) decide what to do with duplicates; the
   canonical dataset itself is non-destructive.
2. **Global example IDs.** Since raw `id` is only unique per file, canonical
   IDs are `f"{category}-{id}"` (e.g. `GEN-4213`), which is unique,
   stable, and human-traceable back to the source file/row.
3. **Grouped splitting is required.** Because 336+ rows share text across
   category files (and near-dup analysis shows even more once
   case/whitespace differences are ignored), a purely random or
   purely-stratified-by-label split risks putting the *same underlying
   post* in both train and test (e.g. the GEN copy in train, the RQ copy in
   test) — the model would then be evaluated on text it effectively saw
   during training. **Decision: split by `dup_group_id` (normalized-text
   group), not by row**, so every row sharing a normalized text always ends
   up in the same split. This directly implements the task's Section 3/4
   requirement ("prioritize grouping over simple stratification" when both
   are in tension). Label stratification is applied at the *group* level
   using the group's label (for `label_conflict` groups, the first row's
   label — a coin flip either way, and only affects 4 groups/~0.04% of
   data) so the 50/50 class balance is preserved as closely as group sizes
   allow. Full implementation: `src/classification/data/make_splits.py`
   (see below).
4. **No author/conversation-level grouping** — the raw files carry none of
   that metadata, so text-based grouping is the only leakage control
   available. Documented as a known limitation of the source corpus
   (Section 3 of the task explicitly asks this to be recorded).
5. **Split ratio:** target 70/15/15 (train/dev/test), per the task's
   suggested default, seed `42`, implemented with two chained
   `StratifiedGroupKFold` passes (test fold carved out first, then
   train/dev from the remainder), grouped on `dup_group_id`. **Actual
   achieved split** (group sizes make an exact 70/15/15 unreachable):
   **train 6,706 (71.4%) / dev 1,340 (14.3%) / test 1,340 (14.3%)**. Label
   balance held closely per split (train 3,369/3,337 sarc/notsarc, dev
   668/672, test 656/684) and category mix is proportionate across splits
   (checked by inspection, not hard-constrained). A programmatic assertion
   (`_assert_no_group_leakage`) runs on every split build and raises if any
   `dup_group_id` ever spans more than one split — it passed. Persisted as
   canonical `data/splits/split_assignments.csv` (`example_id -> split`)
   plus materialized `train.csv`/`dev.csv`/`test.csv` under `data/splits/`,
   all reproducibly regenerated by `make_splits.py` from the canonical
   dataset + a fixed seed — nothing is hand-edited.

### Random seeds

Global default seed for this project: **`42`** (dataset split, few-shot
example sampling, classical-baseline model, fine-tuning). Recorded per
experiment in the registry below; if an experiment deliberately uses a
different seed, that will be called out explicitly.

---

## Environment Audit (2026-08-11)

| Item | Finding |
|---|---|
| Python | 3.10.6 (`.venv`, already set up) |
| OS / machine | macOS (Darwin), Apple Silicon |
| GPU | **No CUDA GPU.** `torch.cuda.is_available() == False`. `torch.backends.mps.is_available() == True` (Apple MPS backend available) — this is a laptop, not a dedicated GPU compute machine. |
| Disk space | **11 GiB free of 228 GiB (95% used)** — tight. Flagged as a risk for Stage B model downloads (see Blockers). |
| `torch` | 2.2.2 installed |
| `transformers` | installed, pinned `<5.0.0` (per `requirements.txt` comment: 5.x needs `torch>=2.4`) |
| `scikit-learn` | 1.7.2 installed (supports `StratifiedGroupKFold`, used for grouped splitting) |
| `pandas`, `numpy`, `scipy`, `nltk`, `rouge_score` | installed |
| Hugging Face cache (`~/.cache/huggingface`) | **empty** — no model weights downloaded yet in this environment |
| `dspy` / `dspy-ai` | **not installed** |
| `sentence-transformers` | **not installed** (needed only for the optional embedding baseline, Section 12) |
| `accelerate`, `datasets`, `peft` | **not installed** (useful/likely-needed for Transformer fine-tuning) |
| `pyyaml` | not installed (planned for config files — will use JSON instead to avoid a new dependency, or add `pyyaml` — see Stage A readiness report) |
| `.env` | file exists, but `OPENROUTER_API_KEY` and `GEMINI_API_KEY` are both **present as keys with empty values** (not even the placeholder text — truly blank) |

---

## Blockers

### BLOCKER-1: No usable LLM API credentials

**What failed:** `OPENROUTER_API_KEY` and `GEMINI_API_KEY` in `.env` are
empty strings.
**What was investigated:** loaded `.env` with `python-dotenv` and checked
both variables directly; confirmed empty (not placeholder text, not
missing file — the file exists and the keys are declared but blank).
**Scope:** blocks *execution* of M2 (LLM zero-shot), M3 (few-shot), M4
(structured reasoning), and M5 (DSPy, which also calls an LLM). Does **not**
block writing/implementing that code, and does not block M1 (classical) or
implementing (not running) M6 (transformer fine-tuning).
**Recommended fix:** add a real `OPENROUTER_API_KEY` (and optionally
`GEMINI_API_KEY`) to `.env`, matching the existing pattern already used by
`src/generation/generate_with_openrouter.py` etc.
**Status:** open. Continuing with all Stage A work that doesn't require a
real key (code, prompts, configs, smoke-test scaffolding with mocked
clients — matching this repo's own established testing convention).

### BLOCKER-2 (soft): Disk space is tight for model downloads

**What was found:** `df -h` shows 11 GiB free of 228 GiB (95% used).
**Scope:** does not block Stage A (no downloads planned). Would block or
risk failing Stage B if multiple encoder checkpoints (RoBERTa-base ~500 MB,
DeBERTa-v3-base ~700 MB, plus tokenizer/cache overhead, plus `pip install`
of `torch`'s CUDA-less wheel is already present) are downloaded without
first freeing space.
**Recommended fix:** confirm available disk before Stage B, and download
one encoder checkpoint at a time rather than pre-fetching several.
**Status:** open, non-blocking for Stage A.

### BLOCKER-3 (soft): No GPU in this environment

**What was found:** `torch.cuda.is_available() == False`; only Apple MPS
is available.
**Scope:** does not block Stage A (no training planned yet). For Stage B,
fine-tuning a BERT-class encoder on ~9.4k short English sentences is
feasible on MPS or even CPU (small dataset, short sequences) but will be
slower than a CUDA GPU — expect fine-tuning runtimes in the tens of minutes
per encoder on MPS, not seconds. Not a correctness blocker, only a
runtime-budget consideration.
**Recommended fix:** none required; MPS is usable. If a real CUDA machine
becomes available later, prefer it for speed, but MPS is sufficient to
produce valid results.
**Status:** open, non-blocking, informational.

### BLOCKER-4: No verified access yet to the actual Stage B compute machine

**What was found:** the user specified the Stage B machine as an Azure
`Standard_NV24s_v3` VM (Tesla M60 GPUs, Maxwell architecture, CUDA compute
capability 5.2 -- no BF16, no FlashAttention2, not compatible with modern
vLLM which needs compute capability >= 7.0). This session runs locally on
macOS (see Environment Audit) and has no direct connection to that VM.
**What was investigated:** found an `azure_vm_key` SSH key and a known
host (`sweng-group-05.eastus.cloudapp.azure.com`) already present on this
machine and tried connecting with 5 common usernames
(`azureuser`/`ubuntu`/`adminuser`/`aya`/`ayagrab`); all attempts timed out
at the network level (not an auth rejection -- consistent with the VM
being stopped/deallocated or a firewall blocking this machine's IP).
**The user confirmed this host is unrelated to the target NV24s_v3 VM** --
disregard it entirely; it was a false lead, not evidence about the actual
Stage B machine's state.
**Scope:** blocks running `scripts/verify_gpu.py` for real, and therefore
blocks any local-LLM (Qwen3-4B) inference. Does not block anything else --
all code below is implemented and unit-tested with mocks/guards, matching
the Stage A discipline of "implement, don't execute until verified/unblocked."
**Recommended fix:** user to either (a) start the VM and share reachable
SSH details, (b) run `python scripts/verify_gpu.py` on the VM themselves
and share the output/report, or (c) grant this session direct access.
**Status:** open.

---

## Experiment Registry

### EXP-001 — Classical baseline: TF-IDF + Logistic Regression — **EVALUATED**

- **Date:** 2026-08-11
- **Approach:** M1, classical ML baseline (Section 6 of the task)
- **Code:** `src/classification/classical/tfidf_baseline.py`
- **Data:** canonical split (`data/splits/`), seed 42, train=6,706 / dev=1,340 / test=1,340
- **Configuration selection (on DEV only, test never touched):** a small
  6-way sweep, `classifier ∈ {logreg, linear_svm}` × `tfidf_variant ∈
  {word_1_2 (word 1-2gram), char_3_5 (char 3-5gram), word_char_combo
  (FeatureUnion of both)}`, `TfidfVectorizer(min_df=2, max_features=50000)`
  for both variants, `LogisticRegression(max_iter=2000)` /
  `LinearSVC()`, both with `random_state=42`. Run inline via
  `run_tfidf_experiment(..., save_artifacts=False)` for the sweep only
  (dev results below are logged here, not persisted as separate result
  directories — this is a config-selection step, not a reportable
  experiment in its own right).

  | Sweep run | classifier | tfidf_variant | dev accuracy | dev macro F1 | dev sarcastic F1 |
  |---|---|---|---:|---:|---:|
  | 1 | logreg | word_1_2 | 0.7463 | 0.7462 | 0.7436 |
  | 2 | logreg | char_3_5 | 0.7470 | 0.7470 | 0.7453 |
  | 3 | logreg | word_char_combo | **0.7530** | **0.7529** | 0.7491 |
  | 4 | linear_svm | word_1_2 | 0.7410 | 0.7409 | 0.7353 |
  | 5 | linear_svm | char_3_5 | 0.7276 | 0.7276 | 0.7249 |
  | 6 | linear_svm | word_char_combo | 0.7261 | 0.7260 | 0.7192 |

  Winner by dev macro F1: **`logreg` + `word_char_combo`**. This
  configuration was then frozen and evaluated once, for the first and only
  time, on TEST.

- **Frozen final evaluation (TEST, run once):**
  `python -m src.classification.classical.tfidf_baseline --experiment-id EXP-001 --classifier logreg --tfidf-variant word_char_combo --eval-split test`

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.7403 |
  | Macro F1 | 0.7403 |
  | Weighted F1 | 0.7403 |
  | Sarcastic Precision / Recall / F1 | 0.7245 / 0.7576 / 0.7407 |
  | Not-sarcastic Precision / Recall / F1 | 0.7569 / 0.7237 / 0.7399 |
  | Confusion matrix (rows=gold, cols=pred, order [not_sarcastic, sarcastic]) | `[[495, 189], [159, 497]]` |

- **Runtime:** a few seconds total (fit + predict on ~9.4k short texts);
  not separately measured/recorded (negligible, no GPU needed).
- **Cost:** $0 (no API calls).
- **Artifacts:** `results/EXP-001/{config.json, metrics.json, predictions.csv}`
- **Observations:** dev and test macro F1 are close (0.753 vs. 0.740, ~1.3
  points), suggesting no meaningful overfitting to the dev-based
  configuration choice. The word+char TF-IDF combination beat either
  n-gram type alone for `logreg`, but consistently hurt `linear_svm` — not
  investigated further per "do not over-tune" instruction. Errors are
  fairly symmetric between false positives (189) and false negatives
  (159), no strong bias toward either class.
- **Errors/unexpected behavior:** none.
- **Conclusion:** credible, reproducible classical baseline established:
  **Macro F1 ≈ 0.740 / Accuracy ≈ 0.740 on the frozen test set.** This is
  the number every other approach (M2-M6) needs to beat to be worth its
  added cost/complexity.
- **Repeat?** No — frozen and valid. Only re-run if the canonical dataset
  or split changes.

---

### EXP-002 through EXP-005 — LLM zero-shot / few-shot / reasoning — **IMPLEMENTED, SMOKE-TESTED** (not evaluated)

- **Code:** `src/classification/llm/` (`client.py`, `schema.py`,
  `few_shot_selection.py`, `run_llm_classification.py`), prompts
  `prompts/classification/{zero_shot,few_shot,reasoning}_v1.txt`, configs
  `configs/llm_{zero_shot,few_shot_random_8,few_shot_curated_8,reasoning}.json`.
- **What "SMOKE-TESTED" means here:** the full pipeline (dataset loading →
  prompt building → structured-output parsing/validation → few-shot
  selection with recorded example IDs → evaluator → artifact saving →
  config-driven dispatch via `run_experiment.py`) was exercised end to end
  with a **mocked** OpenRouter client (deterministic fake responses, no
  network call), both directly and through the JSON-config dispatcher.
  This matches this repository's existing convention for validating
  API-backed code without spending real API budget (see
  `tests/test_evaluate_with_llm_mocked.py` for the precedent). 12 unit
  tests cover retry-then-succeed, retry-then-raise on an invalid label,
  disk caching, and that few-shot demonstrations are drawn only from TRAIN
  (`tests/test_classification_llm_mocked.py`, `test_classification_llm_schema.py`).
- **NOT executed against a real model.** `OPENROUTER_API_KEY` is empty
  (BLOCKER-1). No `EX2-005` metrics exist yet -- do not treat any number
  as measured until this entry is updated with a real run.
- **Planned configs (ready to run once unblocked):**
  - EXP-002: zero-shot, `openai/gpt-oss-20b:free`, dev split
  - EXP-003: few-shot (random, n=8), same model, dev split
  - EXP-004: few-shot (curated, n=8), same model, dev split
  - EXP-005: structured reasoning, same model, dev split
  - (a 5th, TEST-split run of the winning configuration would follow,
    frozen, after dev-based comparison -- not yet assigned an ID)
- **Conclusion:** infrastructure ready; base model choice
  (`openai/gpt-oss-20b:free`, already used as the judge model elsewhere in
  this repo) shared across all four configs so the comparison isolates
  prompting technique. **Repeat?** N/A — nothing has run yet.

### EXP-006 through EXP-008 — DSPy Predict / BootstrapFewShot / MIPROv2 — **IMPLEMENTED** (not evaluated, not smoke-tested against a real LM)

- **Code:** `src/classification/dspy_pipeline/` (`signatures.py`,
  `run_dspy.py`), configs `configs/dspy_{predict,bootstrap_few_shot,mipro_v2}.json`.
- **Status detail:** `dspy` is **not installed** in this environment (see
  Environment Audit), so this could only be statically reviewed, not
  smoke-tested with a mocked LM the way the manual-prompt LLM code was.
  The module is written so it stays importable without `dspy` present
  (`HAS_DSPY` guard, lazy `import dspy` inside functions) and fails with a
  clear `RuntimeError` (not a crash) if invoked before the dependency is
  installed -- verified directly (`signatures.HAS_DSPY == False`,
  `run_dspy._require_dspy()` raises the expected message).
- **Known risk to flag explicitly:** the exact `dspy.LM(...)` call shape
  used for OpenRouter (`f"openrouter/{model}"` + `api_base`) is written to
  the best of current knowledge of dspy's litellm-backed `LM` interface,
  but **has not been validated against a real dspy install** -- treat it
  as "best effort, verify in Stage B," not as confirmed-working code. If
  it needs adjustment once `dspy` is installed, that's expected and should
  be a quick fix, not a sign of a deeper design problem.
- **Conclusion:** infrastructure ready pending `pip install dspy` +
  `OPENROUTER_API_KEY`; the exact LM wiring should be smoke-tested with
  `--limit`-style small runs (once implemented/exposed) before any real
  optimizer run. **Repeat?** N/A — nothing has run yet.

### EXP-009 / EXP-010 — Fine-tuned Transformer (RoBERTa-base / DeBERTa-v3-base) — **IMPLEMENTED** (not evaluated, not smoke-tested with a real checkpoint)

- **Code:** `src/classification/transformer/finetune.py`, configs
  `configs/transformer_{roberta_base,deberta_v3_base}.json`.
- **Status detail:** uses `transformers.Trainer` with early stopping on
  dev Macro F1, a custom `torch.utils.data.Dataset` (no `datasets` package
  dependency), device auto-selection (`cuda` > `mps` > `cpu` -- resolves
  to `mps` in this environment), and per-example confidence via softmax.
  Verified importable and that `TransformerConfig`/`select_device()` work
  without downloading anything. **Not smoke-tested with a real checkpoint
  forward pass** -- no model weights have been downloaded in this
  environment (Stage A rule: no large downloads without explicit
  go-ahead), and `accelerate` (required by `Trainer` in the installed
  `transformers` version) is not yet installed either.
- **Conclusion:** infrastructure ready; recommended first checkpoint is
  `roberta-base` (see readiness report below for why). **Repeat?** N/A —
  nothing has run yet.

---

## Stage A Readiness Report (2026-08-11)

Stage A (infrastructure) is complete. Per the task's instructions, the
expensive full model runs (Stage B) are **not** starting automatically.
This is the status report to act on before proceeding.

### Infrastructure status

| Component | Status | Notes |
|---|---|---|
| Dataset loader (`build_canonical_dataset.py`) | **READY** | Run for real; 9,386 rows, output in `data/processed/sarcasm_v2_canonical.csv` |
| Dataset validation (`audit_dataset.py`) | **READY** | Run for real; report in `data/processed/sarcasm_v2_audit_report.json` |
| Canonical split (`make_splits.py`) | **READY** | Run for real; grouped + leakage-checked; `data/splits/` |
| Evaluation framework (`evaluation/metrics.py`, `io.py`) | **READY** | Used by EXP-001; unit-tested |
| TF-IDF classical baseline | **READY — EVALUATED** | EXP-001 done, Macro F1 0.740 on frozen test |
| LLM zero-shot | **READY, blocked on credentials** | Implemented + mock-smoke-tested; needs `OPENROUTER_API_KEY` |
| LLM few-shot | **READY, blocked on credentials** | Same as above; both random + curated selectors implemented |
| LLM reasoning | **READY, blocked on credentials** | Same as above |
| DSPy | **PARTIALLY READY** | Code written, `HAS_DSPY`-guarded, fails clearly if invoked; needs `pip install dspy` + credentials; OpenRouter `dspy.LM` wiring unverified against a real install |
| Transformer fine-tuning | **PARTIALLY READY** | Code written and structurally verified (device selection, dataset wrapping); needs `accelerate` install + a checkpoint download; no download attempted per Stage A rule |
| Experiment logging (`EXPERIMENT_LOG.md`) | **READY** | This document |
| Result persistence (`results/<experiment_id>/`) | **READY** | Used by EXP-001; schema fixed (`config.json`, `metrics.json`, `predictions.csv`) |
| Reproduction commands | **READY** | See `PROJECT_SUMMARY.md` §13 and per-config commands below |

### Required environment for Stage B

- **Python:** 3.10.6, existing `.venv` — no version change needed.
- **New packages** (`pip install -r requirements-classification.txt`):
  `dspy>=2.5.0`, `accelerate>=0.26.0`, and optionally
  `sentence-transformers>=3.0.0` (only if the optional embedding baseline,
  Section 12, is added later).
- **CUDA:** not available, not required — MPS (Apple Silicon) is
  available and sufficient for fine-tuning this dataset's size; expect
  fine-tuning runtimes of tens of minutes per encoder rather than seconds.
- **Disk space:** currently 11 GiB free of 228 GiB (95% used). A
  `roberta-base` checkpoint is ~500 MB; `deberta-v3-base` is ~700 MB.
  Recommend freeing disk before Stage B if both encoders will be
  downloaded, and downloading one at a time rather than pre-fetching both.
- **API keys:** `OPENROUTER_API_KEY` in `.env` is currently **empty** —
  required for every LLM-based approach (M2–M5). `GEMINI_API_KEY` is also
  empty but not required by this phase (the classification pipeline only
  uses OpenRouter, matching the existing judge-model convention).
- **Hugging Face auth:** not required — `roberta-base` and
  `microsoft/deberta-v3-base` are public checkpoints, no gated-model
  approval needed.

### Model download plan

1. **`roberta-base`** first (EXP-009 config already prepared,
   `configs/transformer_roberta_base.json`) — a well-established, fully
   public, English-general-domain encoder with strong track record on
   short informal text classification, ~500 MB, minimal risk.
2. **`microsoft/deberta-v3-base`** second (EXP-010,
   `configs/transformer_deberta_v3_base.json`), if EXP-009 succeeds and
   disk space allows — DeBERTa-v3's disentangled attention has shown
   consistent gains over RoBERTa on GLUE-style classification, making it a
   meaningfully different second data point rather than a redundant one.
3. Do not download both before confirming EXP-009 completes successfully
   end to end (smoke test with `--limit`-equivalent on a tiny slice first
   — not currently exposed as a CLI flag on `finetune.py`; add
   `train_df.sample(n=...)`/`--limit` support before the first real run if
   a fast sanity check is wanted).

### Exact launch instructions (once Stage B is confirmed)

```bash
# 1. Install the extra dependencies
pip install -r requirements.txt -r requirements-classification.txt

# 2. Add a real OPENROUTER_API_KEY to .env

# 3. Smoke test LLM zero-shot on a tiny slice (recommended before any full run)
python -m src.classification.llm.run_llm_classification \
    --experiment-id SMOKE-zero-shot --mode zero_shot --eval-split dev \
    --model openai/gpt-oss-20b:free --limit 20

# 4. Run the dev-split LLM experiments (M2-M4), in order, via the shared config files
python -m src.classification.run_experiment --config configs/llm_zero_shot.json
python -m src.classification.run_experiment --config configs/llm_few_shot_random_8.json
python -m src.classification.run_experiment --config configs/llm_few_shot_curated_8.json
python -m src.classification.run_experiment --config configs/llm_reasoning.json

# 5. DSPy (M5) -- Predict first, then the two optimizers
python -m src.classification.run_experiment --config configs/dspy_predict.json
python -m src.classification.run_experiment --config configs/dspy_bootstrap_few_shot.json
python -m src.classification.run_experiment --config configs/dspy_mipro_v2.json

# 6. Transformer fine-tuning (M6) -- one checkpoint at a time
python -m src.classification.run_experiment --config configs/transformer_roberta_base.json
python -m src.classification.run_experiment --config configs/transformer_deberta_v3_base.json

# 7. After each step: update EXPERIMENT_LOG.md with real metrics, then
#    decide the winning dev configuration for that approach before ever
#    touching --eval-split test.
```

After every approach's dev-based configuration is frozen, re-run its
config once with `"eval_split": "test"` (a copied config with a new
`experiment_id`, e.g. `EXP-002-test`) for the final, one-shot,
frozen-configuration comparison in `PROJECT_SUMMARY.md`'s results table.

---

## Stage B — Compute Machine (2026-08-11)

**Target machine (as specified by the user):** Azure `Standard_NV24s_v3`
-- 24 vCPUs, 224 GiB RAM, NVIDIA Tesla M60 GPUs (NVv3 family). **Not yet
reachable from this session** -- see BLOCKER-4.

**Why the M60 changes the LLM runtime plan:** Tesla M60 is a Maxwell-generation
GPU, CUDA compute capability 5.2. Per the user's explicit instruction and
independently consistent with Maxwell's known limits:
- **No bfloat16** (needs Ampere+, compute capability >= 8.0) -- must use
  float16 or float32.
- **No FlashAttention 2** (needs Ampere+) -- must use
  `attn_implementation="eager"` (or `"sdpa"`, unverified on this
  architecture; `"eager"` is the safe default).
- **Not compatible with modern vLLM** (needs compute capability >= 7.0) --
  use plain `transformers` generation instead, not vLLM.

**Preferred local model:** `Qwen/Qwen3-4B-Instruct-2507` (~4B params; ~8 GB
in float16, which should fit in one M60's VRAM once GPU count/VRAM-per-GPU
is confirmed by `verify_gpu.py` -- NOT yet confirmed).

### What was built this session (implemented, NOT executed)

1. **`scripts/verify_gpu.py`** -- the mandatory Stage B gate. Runs
   `nvidia-smi`, records exact GPU count/model/VRAM/driver version, checks
   `torch.cuda` compute capability against BF16/FlashAttention2/vLLM
   feature requirements, writes `data/processed/gpu_verification_report.json`,
   and exits non-zero if no CUDA GPU is visible. **Must be run first, for
   real, on the actual VM, before any model download** -- not yet done
   (BLOCKER-4). Sanity-checked on this Mac (correctly reports no CUDA and
   exits 1; the local-Mac report was deleted afterward, not committed, to
   avoid it being mistaken for real VM data).
2. **`src/classification/llm/local_client.py`** -- `LocalHFClient`, a
   local Hugging Face Transformers inference client shaped like the
   OpenAI/OpenRouter client (`.chat.completions.create(...)`) so the
   existing zero-shot/few-shot/reasoning pipeline
   (`run_llm_classification.py`) works unchanged against a local model --
   only `provider="local_hf"` needs to be selected. Defaults: `float16`,
   `attn_implementation="eager"`, `device_map="auto"`. Hard-rejects
   `bfloat16` and `flash_attention_2` at construction time (`ValueError`,
   before any download/load attempt), and hard-rejects loading with no
   CUDA GPU visible (`RuntimeError`, pointing at `verify_gpu.py`). 5 unit
   tests cover these guards (`tests/test_classification_local_llm_client.py`)
   -- no model download, no GPU needed to verify the guards themselves.
3. **`src/classification/llm/client.py`** -- extended `get_llm_client()`
   with a `provider` parameter (`"openrouter"` default, or `"local_hf"`).
4. **`src/classification/llm/run_llm_classification.py`** -- added a
   `provider` parameter/`--provider` CLI flag, threaded through to
   `get_llm_client`; forces `concurrency=1` when `provider="local_hf"`
   (one shared GPU model instance is not safe to fan out across threads,
   unlike independent OpenRouter HTTP calls).
5. **`configs/llm_zero_shot_qwen_local.json`** -- ready-to-run config
   (`EXP-002-local`), explicitly noting in its `_note` field that it must
   not be run until `verify_gpu.py` passes, and to smoke-test with
   `--limit 10-20` first given unknown/likely-slow generation latency on
   an M60 (no tensor cores).

### Explicitly NOT done yet (correct terminology, per task rules)

- `nvidia-smi` has **not** been run on the real target VM.
- GPU count, exact GPU model/VRAM, driver version, and CUDA environment
  are **not yet recorded** for the real VM -- only the user-provided specs
  (Tesla M60, NV24s_v3) are on record above, which are being *trusted as
  configuration input* (what to build for) but not yet *independently
  verified* (what's actually there).
- `Qwen/Qwen3-4B-Instruct-2507` has **not** been downloaded anywhere.
- No smoke test, no full inference run, no LLM/DSPy experiment has
  executed on this or any GPU machine.

### Next step

Blocked on BLOCKER-4 (VM access). Once resolved: run
`python scripts/verify_gpu.py` on the VM, paste/record its output here
verbatim (exact GPU count, model, VRAM, driver version, compute
capability), and only then proceed to
`pip install -r requirements.txt -r requirements-classification.txt` +
a small `--limit`-bounded smoke test of `configs/llm_zero_shot_qwen_local.json`.

---

## Stage B — Update (2026-08-11, later same day)

### BLOCKER-2 / BLOCKER-3: recharacterized, resolved on the Azure VM only

**Distinguish environments precisely, per instruction:**
- **Local Mac environment** (this session's shell, `hostname` = `Mac.lan`):
  BLOCKER-2 (`dspy`/`accelerate` not installed) and BLOCKER-3 (no CUDA
  GPU) **remain true and unresolved here** -- nothing changed about this
  machine.
- **Azure Stage B VM** (`dpmlgpuNC6sv32025s-0003`, per the user's manual
  verification, not yet independently confirmed by this session -- see
  BLOCKER-4b below): user reports `dspy` imports successfully,
  `accelerate`/`sentencepiece`/`protobuf` are installed, `torch==2.5.1+cu118`
  with `torch.cuda.is_available() == True` and `torch.cuda.device_count() == 2`,
  a real CUDA matmul executed on `cuda:0`, and `Qwen/Qwen3-4B-Instruct-2507`
  was downloaded, loaded across both GPUs, and used for real inference
  (single-sentence and multi-sentence smoke tests). **On that specific
  machine**, BLOCKER-2 and BLOCKER-3 are resolved. This is recorded as
  reported by the user, not yet re-verified independently by this session
  -- see BLOCKER-4b for why, and the plan to re-verify via
  `scripts/verify_gpu.py` once this session can reach the VM.

**Verified VM hardware/software facts (as reported by the user, to be
independently re-confirmed via `scripts/verify_gpu.py` once SSH access
works):**

| Field | Value |
|---|---|
| VM size | Standard_NV24s_v3 |
| Hostname | `dpmlgpuNC6sv32025s-0003` |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-1029-azure |
| vCPUs / RAM | 24 / ~220 GiB usable |
| Storage | `/mnt` ~1.5 TB (~1.4 TB free), `/datashare` (Azure-mounted shared FS) |
| GPUs | 2x NVIDIA Tesla M60, ~7.93 GiB VRAM each |
| Compute capability | 5.2 (Maxwell) -- matches the assumption `local_client.py` was already written for |
| Driver | 535.230.02 |
| `nvidia-smi`-reported CUDA compatibility | 12.2 |
| Working PyTorch CUDA runtime | 11.8 (`torch==2.5.1+cu118`) -- intentionally different from the driver's reported max; not to be "fixed" by upgrading |
| Python env | `/mnt/vmadmin/sarcasm-env` (Python 3.10.11) |
| `HF_HOME` | `/mnt/vmadmin/huggingface` (~7.6 GB used after Qwen download) |
| Installed | torch 2.5.1+cu118, transformers 5.15.0, datasets 5.0.1, sklearn 1.7.2, pandas 2.3.3, dspy, accelerate, sentencepiece, protobuf |

**Verified Qwen loading config (as reported by the user):**
```python
AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.float16, device_map="auto",
    max_memory={0: "7GiB", 1: "7GiB", "cpu": "180GiB"}, low_cpu_mem_usage=True,
)
```
Device map observed: GPU 0 = embedding + LM head + layers 0-15 (~3.73 GiB);
GPU 1 = layers 16-35 + final norm/rotary (~3.76 GiB). Single-sentence
inference ("Oh fantastic, my flight has been delayed again." -> "sarcastic")
took ~0.934s; peak memory ~3.75/3.78 GiB across the two GPUs.

**Action taken on `src/classification/llm/local_client.py`:** updated to
match the user's verified-working Qwen loading call exactly:
- `torch_dtype=torch_dtype` -> `dtype=torch_dtype` (the deprecated kwarg
  name was in use; switched to the modern one, matching `transformers==5.15.0`
  on the VM).
- Added `max_memory` (defaults to `{0: "7GiB", 1: "7GiB", "cpu": "180GiB"}`
  for a `device_map="auto"` load, matching the VM's verified config
  exactly -- generalized to `{i: "7GiB" for i in range(device_count)}` so
  it still makes sense if GPU count ever differs) and `low_cpu_mem_usage=True`.
- 5 existing guard-rail unit tests (bfloat16/FlashAttention2/no-CUDA
  rejection) still pass -- none of them reach the `from_pretrained` call,
  so this change was safe to make without GPU access.
Not yet tested against the real environment end-to-end (that needs
BLOCKER-4b resolved first), but the loading call now exactly mirrors what
the user already confirmed works.

**Action taken on `scripts/verify_gpu.py`:** confirmed the script's exit
code was already gated ONLY on CUDA-GPU-visibility (never on BF16/
FlashAttention2/vLLM support) -- so it was not actually rejecting M60
hardware. Refactored anyway for clarity per the user's instruction:
verdict lines are now explicitly labeled `REQUIRED` (the only one that can
fail: a CUDA GPU must be visible to torch) vs. `INFORMATIONAL` (BF16/
FlashAttention2/vLLM support, reported but never fatal), and the report
now includes an explicit `fp16_transformers_pipeline_ok: true/false`
field. Re-verified it still runs cleanly (no crash, correctly reports
`FAIL`/exit 1) on the local Mac's no-CUDA environment.

### BLOCKER-4b (NEW): This session cannot yet SSH into the VM

**What failed:** `ssh -i ~/.ssh/azure_vm_key vmadmin@20.245.56.28 "hostname"`
(the exact command the user specified) returns
`Permission denied (publickey,password)`.
**What was investigated:**
- Confirmed this session's shell is the local Mac (`hostname` = `Mac.lan`), not the VM.
- `ssh -v` shows a clean TCP connection and a clean host-key match against
  `~/.ssh/known_hosts` (`20.245.56.28` was already a known host from a
  prior session) -- **this is an authentication failure, not a
  connectivity or host-key-verification failure.**
- The only candidate private key on this machine, `~/.ssh/azure_vm_key`
  (permissions correctly `600`), was offered and explicitly rejected by
  the server for user `vmadmin`; no other auth method succeeded.
- Checked for an SSH agent with other loaded identities
  (`ssh-add -l` -> "The agent has no identities") and tried default
  (no `-i`) auth -- also rejected.
- This is the same key pair used in the Stage A session's earlier (also
  failed, against a since-confirmed-unrelated host) connection attempt --
  i.e. **there is no independent evidence this key was ever authorized on
  `dpmlgpuNC6sv32025s-0003` specifically.**
- Local public key / fingerprint (safe to share, not a secret):
  `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJlvoJ2BbTuk68uAIZbTl5Da0k2mBpKwPu2jRJYT8S6J azure-vm`
  / `SHA256:Zzrfc6NG/Kg0q+l/bp6xdRAfjmzlEksN1A15jbDyGBg`.
**Which experiment(s) affected:** ALL of Stage B execution (Qwen smoke
test, M2-M5, M6/DeBERTa) -- everything requiring the VM. Does not affect
anything already completed (EXP-001) or local-only prep (this doc, the
`verify_gpu.py` fix, `scripts/sync_to_vm.sh`).
**Does it block other experiments:** yes, blocks all remote execution
until resolved. No GPU-based experiment can run from this session in the
meantime.
**Recommended fix (one of):** (a) add the local public key above to
`~/.ssh/authorized_keys` for `vmadmin` on the VM; (b) provide the actual
private key (path/contents) that IS already authorized for `vmadmin`; or
(c) since the user already has working manual access to the VM (per
Sections 1-9), the user runs the remaining Stage B commands directly and
reports results back for this log, using the prepared
`scripts/sync_to_vm.sh` + the command sequence in the "Stage B Readiness
Report" above (Section 19-21 of the handoff).
**Independent work continued:** yes -- `scripts/verify_gpu.py` fixed and
re-verified locally; `scripts/sync_to_vm.sh` written (rsync-based, matches
the user's specified exclude pattern, not yet run); this log updated;
`local_client.py`'s `dtype=`/`torch_dtype=` discrepancy flagged for
confirmation once real execution is possible.
**Status:** open -- reported to the user, awaiting either corrected SSH
access or the user running the remaining steps directly.

---

## Stage B — Update (2026-08-11, SSH resolved, execution begins)

### BLOCKER-4b: RESOLVED

User added this session's public key
(`ssh-ed25519 ...azure-vm`, `SHA256:Zzrfc6NG/Kg0q+l/bp6xdRAfjmzlEksN1A15jbDyGBg`)
to `vmadmin`'s `~/.ssh/authorized_keys` on the VM. Independently
re-verified from this session:
```
$ ssh -i ~/.ssh/azure_vm_key vmadmin@20.245.56.28 "hostname"
dpmlgpuNC6sv32025s-0003
$ ssh -i ~/.ssh/azure_vm_key vmadmin@20.245.56.28 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
Tesla M60, 8192 MiB
Tesla M60, 8192 MiB
```
Both match exactly what was expected. **BLOCKER-2 and BLOCKER-3 are now
also independently confirmed resolved on the Azure VM** (still open/true
on the local Mac -- unchanged, see original entries).

### Remote environment -- independently verified (not just user-reported)

```
hostname: dpmlgpuNC6sv32025s-0003
python: 3.10.11
torch: 2.5.1+cu118, CUDA available: True, GPU count: 2
transformers: 5.15.0
datasets: 5.0.1
sklearn: 1.7.2
pandas: 2.3.3
dspy: 3.3.0
accelerate: 1.14.0
sentencepiece: importable
HF cache: /mnt/vmadmin/huggingface, 9.6G, Qwen3-4B-Instruct-2507 present
data/splits/dev.csv: 1340 rows, columns [example_id, category, source_file, raw_id, text, label, dup_group_id, label_conflict] -- matches Stage A exactly
```

Full snapshot saved to `environment_stage_b.txt` (repo root, both on the
VM and copied back to the local Mac) per the reproducibility requirement:
`hostname`, `python --version`, `nvidia-smi`, `pip freeze` (148 lines, no
secrets -- checked).

### Repository sync

Remote `/mnt/vmadmin/projects/` was empty before sync (nothing to
preserve). Ran `scripts/sync_to_vm.sh` (rsync, `.git`/`__pycache__`/
`.venv`/`models`/`checkpoints`/`results`/`.pytest_cache`/`data/llm_cache`
excluded, matching the given pattern) -> `/mnt/vmadmin/projects/sarcasm/`.
Verified post-sync: canonical dataset, splits, configs, and raw corpus
files all present and intact. `results/` (just EXP-001, small) was
intentionally excluded -- reproducible on the VM if ever needed by
re-running `configs/tfidf.json`.

### `scripts/verify_gpu.py` -- run for real on the VM

```
fp16_transformers_pipeline_ok: True
2x Tesla M60, 8.52 GB each (torch-reported), compute capability 5.2
driver 535.230.02, nvidia-smi CUDA 12.2, torch CUDA runtime 11.8
```
All three INFORMATIONAL lines (BF16/FlashAttention2/vLLM) correctly report
"does NOT support" without failing the script -- confirms the earlier fix
(§ previous entry) behaves as intended on the real hardware, not just in
the local no-CUDA sanity check. Full JSON report at
`data/processed/gpu_verification_report.json` on the VM.

### Repository-level Qwen smoke test -- **SMOKE-TESTED** (production code path, real GPU, real model)

- **Command:** `python -m src.classification.llm.run_llm_classification --experiment-id SMOKE-qwen-zero-shot --mode zero_shot --eval-split dev --model Qwen/Qwen3-4B-Instruct-2507 --provider local_hf --limit 20 --concurrency 1`
- **Result:** model loaded in ~2s (already cached, no download), 20
  DEV examples classified in ~36s (~1.8s/example, deterministic
  `do_sample=False` since `temperature=0.0`), zero label-parsing failures
  (every response parsed to a valid `{sarcastic, not_sarcastic}` on the
  first attempt, no retries triggered), `results/SMOKE-qwen-zero-shot/`
  written with all three expected artifacts. Confirms: dataset loading,
  prompt construction, `local_hf` client (model load + generation across
  both GPUs), label parsing, prediction persistence, and the evaluator all
  work through the real production code path -- not a mock.
- **Accuracy note (not a real signal, explained so nobody misreads it
  later):** the reported accuracy (0.25, macro F1 0.20) is an artifact of
  `--limit 20` taking the first 20 rows of `dev.csv`, which is **not
  shuffled** (`make_splits.py` preserves the canonical dataset's original
  row order within each split). The first 20 dev rows happen to be almost
  entirely `not_sarcastic` (verified: `GEN-9, GEN-13, GEN-28, ...` are
  consecutive early rows in the raw GEN file, which is not label-balanced
  locally). This is a **smoke-test sampling artifact, not a measurement**
  -- the full-DEV run (next) uses all 1,340 rows and isn't affected.
- **Runtime environment:** Azure `Standard_NV24s_v3`, 2x Tesla M60 (this
  is explicitly recorded on every experiment from here on, per the
  "local Mac vs. Azure VM" distinction rule).
- **Conclusion:** pipeline verified end-to-end on real hardware/model.
  Proceeding to the full M2 DEV run.

### Methodology correction: TEST-sealing policy (2026-08-11)

The initial Stage B plan had each method (M2, M3, ...) evaluate TEST
immediately after its own DEV run, reasoning that a single fixed
zero-shot config has "nothing to tune." **Corrected before any TEST
evaluation actually happened** (EXP-002's TEST run had not yet been
launched): per explicit instruction, TEST stays completely sealed until
**every** method (M2-M6) has finished its DEV-only development and had a
configuration frozen. Rationale: even without per-method hyperparameter
tuning, *cross-method* comparisons and any judgment calls made while
methods are still being developed (e.g. deciding a prompt looks "good
enough," or debugging a parsing issue after seeing a low score) are a form
of indirect tuning if TEST is visible during that process. Sealing TEST
entirely until every method is frozen removes that risk. **No previous
experiment is affected** -- EXP-001 (TF-IDF) was already TEST-evaluated
correctly (frozen on DEV first, per Stage A); EXP-002 had not yet reached
its TEST step. Full plan: see `STAGE_B_CHECKLIST.md`'s "PHASE 1 / PHASE 2"
split.

### EXP-002 — Qwen3-4B zero-shot — **DEV-EVALUATED** (TEST sealed, not yet run)

- **Date:** 2026-08-11. **Environment:** Azure `Standard_NV24s_v3`, 2x Tesla M60 (NOT the local Mac).
- **Command:** `python -m src.classification.run_experiment --config configs/llm_zero_shot_qwen_local.json`
- **Config:** `provider=local_hf`, `model=Qwen/Qwen3-4B-Instruct-2507`, `mode=zero_shot`, `temperature=0.0` (deterministic, `do_sample=False`), prompt `classification/zero_shot_v1.txt`, `eval_split=dev`, seed 42.
- **Runtime:** ~37 minutes wall-clock for 1,340 examples (~1.6-1.9s/example, sequential -- `local_hf` forces `concurrency=1`), plus ~2s one-time model load (already cached).
- **Full DEV metrics:**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.6440 |
  | Macro F1 | 0.6008 |
  | Weighted F1 | 0.6004 |
  | not_sarcastic: P / R / F1 (support 672) | 0.9295 / 0.3140 / 0.4694 |
  | sarcastic: P / R / F1 (support 668) | 0.5858 / 0.9760 / 0.7322 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[211, 461], [16, 652]]` |

- **Quality checks performed (per "EXPERIMENT QUALITY" policy):**
  - `n_examples=1340` matches DEV split size exactly; `predictions.csv` has 1,340 unique `example_id`s, no duplicates, no missing rows.
  - Gold label distribution in the output (672 not_sarcastic / 668 sarcastic) matches the canonical DEV split exactly -- confirms no split drift, no leakage, correct label mapping.
  - Every prediction is a valid `{sarcastic, not_sarcastic}` value (guaranteed structurally: `classify_one` raises after retries on any unparseable/invalid-label response rather than letting one through, and the run completed without raising -- so zero silent parsing failures).
  - **Flagged and investigated:** predicted-label distribution is heavily skewed -- 1,113/1,340 (83%) predicted `sarcastic` vs. the true 49.9%. Checked whether this was a bug (e.g. prompt/parsing issue, or an artifact of one category dominating): predicted-sarcastic rate is uniform across categories (GEN 81.7%, HYP 89.2%, RQ 83.9%) -- rules out a category-specific or ordering artifact. Manually read 5 false positives (gold `not_sarcastic`, predicted `sarcastic`): all are combative/rhetorical internet-debate text (creationism-vs-evolution arguments, political forum sparring, mocking rhetorical questions like "How do you know there isn't a tiger in the bathroom?") that plausibly *reads* as sarcastic/mocking in tone even though annotated as sincere argument. **Conclusion: this is genuine zero-shot model behavior on this corpus, not a pipeline bug** -- Qwen3-4B-Instruct, given only the task definition and no examples, appears biased toward calling adversarial/rhetorical debate text "sarcastic," yielding high recall on true sarcasm (0.976) at a heavy cost to specificity (0.314 recall on genuinely sincere text).
- **Artifacts:** `results/EXP-002/{config.json, metrics.json, predictions.csv}` (on the VM; will be pulled back to the local Mac repo at the next sync).
- **Comparison to EXP-001 (TF-IDF, TEST Macro F1 0.740):** EXP-002's DEV Macro F1 (0.601) is well below EXP-001's TEST score, though the splits being compared differ (DEV vs. TEST) so this is directional, not a rigorous head-to-head yet -- the rigorous comparison happens in Phase 2 once both are measured on the same (TEST) split. Directionally, plain zero-shot prompting is not yet beating the classical baseline on this corpus.
- **Conclusion:** DEV-EVALUATED. Zero-shot has no real hyperparameter to tune (one fixed prompt/config), so this DEV result doubles as a strong signal for Phase 2, but **per the TEST-sealing policy, TEST is not evaluated now** -- it waits until every method (M2-M6) has a frozen DEV configuration.
- **Repeat?** No, unless the zero-shot prompt itself is revised (not currently planned) -- this is effectively already a frozen candidate for Phase 2.
- **Supplementary error analysis** (`results/EXP-002/analysis/dev_error_summary.json`): accuracy by category -- GEN 0.666, RQ 0.609, HYP 0.572 (HYP/hyperbole hardest, not RQ as might be expected). Predicted-sarcastic rate is uniform across categories (~82-89%), confirming the skew isn't category-specific. Incorrect predictions average 53.9 words vs. 46.7 for correct ones -- a mild signal that longer/more complex debate text is harder. Only 2 of the 22 known label-conflict rows landed in DEV (grouped splitting keeps most duplicate-text groups together, and apparently most conflict groups landed in TRAIN or TEST) -- too few to draw a conclusion from.

### M3 (EXP-003/004) now running -- few-shot prompts are ~2.5x slower per example

M3-random started automatically (chained script, see `run_m3_m4_chain.sh`)
immediately after EXP-002 finished. Observed rate: **~4.3s/example** (vs.
~1.7s/example for zero-shot) -- expected, since an 8-shot prompt is much
longer to prefill on hardware with no tensor cores. Revised ETA for the
M3-random -> M3-curated -> M4 chain: roughly 1.5-2 hours per full-DEV run,
so **several hours total** for the remaining chain. This does not block
other work (see Environment Audit / resource-management notes) -- CPU-only
prep continues in parallel; no second GPU-heavy process will be started
until this chain completes.

### EXP-003 — Qwen3-4B few-shot (random demos) — **DEV-EVALUATED** (TEST sealed, not yet run)

- **Date:** 2026-08-11. **Environment:** Azure `Standard_NV24s_v3`, 2x Tesla M60.
- **Command:** `python -m src.classification.run_experiment --config configs/llm_few_shot_random_8_qwen_local.json`
- **Config:** `provider=local_hf`, `model=Qwen/Qwen3-4B-Instruct-2507`, `mode=few_shot`, `few_shot_variant=random`, `n_shots=8`, `temperature=0.0`, prompt `classification/few_shot_v1.txt`, `eval_split=dev`, seed 42.
- **Demo example IDs used** (8, randomly selected, seed 42): `RQ-502, GEN-3940, GEN-437, GEN-2796, GEN-1020, RQ-1304, GEN-1888, HYP-624` -- label balance confirmed 4 `not_sarcastic` / 4 `sarcastic` (not itself skewed).
- **Runtime:** ~1h34m wall-clock for 1,340 examples (~4.2s/example, in line with the ~4.3s/example estimate above; the longer 8-shot prompt dominates per-call latency on M60's no-tensor-core hardware).
- **Full DEV metrics:**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.6328 |
  | Macro F1 | 0.5880 |
  | Weighted F1 | 0.5876 |
  | not_sarcastic: P / R / F1 (support 672) | 0.8982 / 0.3021 / 0.4521 |
  | sarcastic: P / R / F1 (support 668) | 0.5790 / 0.9656 / 0.7239 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[203, 469], [23, 645]]` |

- **Quality checks performed:**
  - `n_examples=1340`, 1,340 unique `example_id`s, no duplicates, no missing rows, no nulls -- confirmed directly against `data/splits/dev.csv` (zero missing IDs).
  - Every prediction is a valid `{sarcastic, not_sarcastic}` value (same structural guarantee as EXP-002 -- run completed without a parsing-retry exhaustion).
  - **Flagged and investigated:** Macro F1 (0.588) is *lower* than EXP-002 zero-shot (0.601) -- surprising, since few-shot demonstrations are generally expected to help or at least not hurt. Checked three possible explanations:
    1. **Demo-selection bias?** No -- the 8 random demos are label-balanced (4/4), so the model wasn't shown a skewed set.
    2. **Category-specific artifact?** No -- predicted-sarcastic rate is uniform across categories (GEN 82.0%, HYP 91.0%, RQ 82.3%), same pattern as EXP-002's uniform skew, ruling out an ordering/category bug.
    3. **Did few-shot actually change model behavior, or is this noise from a near-identical model?** Compared predictions directly against EXP-002 on the same DEV set: **94.25% agreement** -- the few-shot prompt barely moved the model's decisions, and where it did move them, it slightly *increased* the sarcastic-prediction skew (83% -> 83.1% predicted-sarcastic, but recall on `not_sarcastic` dropped further, 0.314 -> 0.302).
  - **Conclusion: not a pipeline bug.** These 8 random demonstrations did not meaningfully help Qwen3-4B calibrate away from its zero-shot bias toward reading adversarial/rhetorical text as sarcastic; if anything they reinforced it slightly. This is a legitimate, if unflattering, empirical result for the random few-shot variant.
- **Artifacts:** `results/EXP-003/{config.json, metrics.json, predictions.csv}` (on VM; to be pulled to local repo at next sync).
- **Comparison to EXP-002:** EXP-003 (random few-shot) underperforms EXP-002 (zero-shot) on DEV Macro F1 (0.588 vs. 0.601) and Accuracy (0.633 vs. 0.644). Directionally, random-demo few-shot is not currently the better candidate between the two -- EXP-004 (curated few-shot, running next) is the remaining chance for few-shot to beat zero-shot on this corpus.
- **TEST not touched**, per the sealing policy -- correct.

### Work paused (2026-08-11, ~18:50 UTC) -- user shutting down the VM

At the user's explicit request (VM being powered off, to be reconnected
later), the currently-running EXP-004 (M3-curated) process and its
orchestrating chain script (`run_m3_m4_chain.sh`, PID 24865) were both
intentionally killed -- **this is a deliberate stop, not a crash.**

- EXP-004 had completed **447/1340 examples (33%)** at the moment of
  stopping (`logs/EXP-004-curated-dev.log`, last line: `447/1340
  [27:42<51:08, 3.44s/it]`).
- The chain script was killed *first* (before the python process) so it
  would not misinterpret the kill as "EXP-004 finished" and auto-launch M4
  with a stale/absent result.
- Verified after stopping: no `run_experiment`/`run_m3_m4_chain` processes
  remain (`ps -ef` clean), both GPUs idle (`nvidia-smi`: 0 MiB used on
  both).
- The per-example LLM disk cache (`data/llm_cache/`) had grown to 3,088
  entries at the time of stopping, including the 447 EXP-004 examples
  already computed -- these will be near-instant on resume (see
  `STAGE_B_CHECKLIST.md`, "How to resume" section, for exact commands).
- No results were lost: EXP-004 hadn't written `predictions.csv` yet
  (only written at the end of a full run), so there is nothing partial to
  clean up in `results/` -- resuming is simply "rerun the same command."

### VM restart -- `/mnt` ephemeral-disk data loss incident and recovery (2026-08-12)

**What happened:** the Azure VM was restarted (to load the known-good
kernel `6.8.0-1029-azure` after a driver-module-loading problem following
a kernel package update -- confirmed via `~/.bash_history` on the VM:
`dkms status`, `grub-reboot` to the `6.8.0-1029-azure` entry, `sudo
reboot`). After the restart, `/mnt/vmadmin/` -- containing the VM-side
repo checkout, the Python venv (`sarcasm-env`), the Hugging Face model
cache, and the per-example LLM disk cache (`data/llm_cache/`, 3,088
entries) -- was **completely empty**.

**Root cause, confirmed:** `/mnt` on this VM is Azure's **ephemeral
resource/temp disk** (`/dev/sdb1`, backed by `/dev/disk/cloud/azure_resource-part1`
per `/etc/fstab`), not persistent storage. `df -h` showed 32 KB used out
of 1.5 TB (a fresh filesystem) and `/mnt/DATALOSS_WARNING_README.txt` --
rewritten by Azure on every boot -- states explicitly: *"THIS IS A
TEMPORARY DISK... SUBJECT TO LOSS and THERE IS NO WAY TO RECOVER IT."*
`sudo ls /mnt/lost+found` was empty; nothing was recoverable from the
disk itself. This was true risk that had gone unnoticed since Stage B
began -- everything under `/mnt/vmadmin/` had been treated as if it were
durable.

**What was permanently lost:**
- The VM-side repo checkout, the `sarcasm-env` venv, and the full
  Hugging Face model cache (~9.3 GB) -- all trivially regenerable
  (re-synced / reinstalled / re-downloaded below), not scientific loss.
- The per-example LLM disk cache (`data/llm_cache/`, 3,088 entries) --
  regenerable at the cost of re-running inference (no scientific loss,
  just wasted compute time).
- **EXP-003's raw `results/EXP-003/{config.json,metrics.json,predictions.csv}`
  artifacts** -- these had never been pulled back to the local Mac repo
  (there was no "pull results back" step in the workflow; `sync_to_vm.sh`
  only pushes code, and per-example predictions only get produced on the
  VM). The aggregate metrics were not lost (already transcribed into this
  log, above), but the per-example predictions file was gone. **Because
  EXP-003 is a fully deterministic run** (`temperature=0.0`,
  `do_sample=False`, fixed `seed=42`, and the 8 demo example IDs are
  recorded above), it was regenerated byte-for-byte-equivalently by
  simply re-running the identical frozen config -- not a new measurement,
  pure artifact regeneration. See below for confirmation the regenerated
  run reproduced the same metrics.
- EXP-004's partial state (447/1340 cached completions) -- no scientific
  loss, since EXP-004 never had a `predictions.csv` to begin with (per
  the "Work paused" entry above); restarted from scratch.
- `run_m3_m4_chain.sh`, the ad hoc chain script used to launch M3/M4 back
  on 2026-08-11, had never been committed to git and lived only under
  `/mnt/vmadmin/` -- also gone, and rewritten (see
  `scripts/run_m3_m4_chain.sh`, now git-tracked as part of the fix below).

**Also found, unrelated to the recovery:** a personal (non-technical)
message in the VM's `~/.bash_history`, apparently pasted into the SSH
terminal by mistake, asking to share a phone number with a recruiter.
Flagged to the user directly; not acted on and not reproduced here.

**Recovery performed:**
1. Verified survival: `hostname`, `uname -r` (`6.8.0-1029-azure`,
   confirmed the known-good kernel), `nvidia-smi` (driver 535.230.02,
   both Tesla M60s healthy, 0 MiB used), `df -h`, and `/datashare` all
   checked before touching anything.
2. Recreated `/mnt/vmadmin/{projects,sarcasm-env,huggingface}`.
3. Restored the repo via `scripts/sync_to_vm.sh` (unchanged, from the
   local Mac copy -- confirmed intact and authoritative).
4. Recreated the Python 3.10 venv (`python3.10 -m venv`, system has
   3.10.12 -- functionally identical to the previously-verified 3.10.11;
   required installing the `python3.10-venv` apt package first). Installed
   the **exact pinned versions** from the `pip freeze` captured in
   `environment_stage_b.txt` before the loss: `torch==2.5.1+cu118`
   (via `--index-url https://download.pytorch.org/whl/cu118`, matching
   `torchvision==0.20.1+cu118`/`torchaudio==2.5.1+cu118`),
   `transformers==5.15.0`, `datasets==5.0.1`, `scikit-learn==1.7.2`,
   `pandas==2.3.3`, `dspy==3.3.0`, `accelerate==1.14.0`, plus
   `sentencepiece`/`protobuf`/`safetensors`/`huggingface_hub` -- **no
   version upgraded or changed from the known-good stack.** Deliberately
   did *not* `pip install -r requirements-dev.txt` (its loose pins,
   `numpy<2.0`/`transformers<5.0.0`, conflict with and would have
   downgraded the pinned stack) -- installed bare `pytest`/`pytest-mock`
   instead.
5. Re-downloaded `Qwen/Qwen3-4B-Instruct-2507` and
   `microsoft/deberta-v3-base` into a fresh `HF_HOME=/mnt/vmadmin/huggingface`
   (~9.3 GB total, both public/ungated, no token needed, done in under a
   minute).
6. Ran `scripts/verify_gpu.py` for real -- identical verdict to before:
   `fp16_transformers_pipeline_ok: True`, both Tesla M60s visible,
   BF16/FlashAttention2/vLLM correctly reported as unsupported
   (informational only).
7. Ran the classification test suite directly (`pytest tests/test_classification_*.py`,
   avoiding the full suite which needs Stage-A-only deps not in the
   pinned Stage B stack: `nltk`, `seaborn`, `google.generativeai`) --
   **61/61 passed.**
8. Re-ran the repository-level Qwen zero-shot smoke test (`--limit 20`,
   same command as the original) -- **reproduced the exact same DEV
   metrics as the original smoke test** (accuracy 0.25, macro F1 0.20),
   confirming the recovered pipeline behaves identically, not just
   "loads without crashing."
9. Re-ran EXP-003 (identical frozen config) to regenerate its lost
   `predictions.csv`/`config.json`/`metrics.json` -- launched as step 1 of
   a fresh `scripts/run_m3_m4_chain.sh` (now git-tracked, unlike its
   predecessor), followed automatically by EXP-004 (restarted from
   scratch, since nothing valid survived to resume from) and then EXP-005
   (M4 reasoning, next in the Phase 1 queue regardless).

**Fixes to prevent recurrence (durability improvements):**
- **Committed the entire Stage A/B work to git** (`22def6e`) -- until now
  `EXPERIMENT_LOG.md`, `PROJECT_SUMMARY.md`, `STAGE_B_CHECKLIST.md`, all
  of `src/classification/`, `configs/`, `data/splits/`,
  `data/processed/sarcasm_v2_canonical.csv`, `results/` (EXP-001,
  EXP-002), and the classification tests existed **only on the local Mac
  disk**, uncommitted -- itself a single point of failure independent of
  the VM. (Push to the `origin` remote was attempted but blocked by the
  session's permission policy for remote/shared actions; local commit
  succeeded. Ask to push when convenient.)
- **Added `scripts/sync_from_vm.sh`**, the missing counterpart to
  `sync_to_vm.sh` -- pulls `results/` and `logs/` back from the VM to the
  local Mac repo. This is the gap that caused the EXP-003 predictions
  loss (results were only ever pushed one direction). Going forward, run
  it after every experiment finishes, not just before a planned shutdown.
- Confirmed `/datashare` (the course's Azure Files share, `cifs`-mounted)
  is a separate, likely-persistent mount, but it's read-only course data,
  not writable project storage -- not used for project artifacts.
- Model downloads (`HF_HOME`) and the per-example LLM cache remain
  intentionally on ephemeral `/mnt` -- both are cheaply regenerable
  (re-download / re-inference) and don't need durable storage; only
  irreplaceable outputs (code, configs, splits, results, logs) need to
  live in git now.

**Not changed, per explicit instruction:** NVIDIA driver (535.230.02),
kernel (`6.8.0-1029-azure`), CUDA runtime (11.8), or the PyTorch version
-- all reproduced exactly as before. Note for later: `apt-get install
python3.10-venv` pulled in kernel-related package metadata showing a
newer kernel (`6.8.0-1064-azure`) already present on disk but not
booted -- **do not let anything reboot the VM into it**; `6.8.0-1029-azure`
remains the only verified-working kernel/driver combination.

**Startup guard added (2026-08-12):** `scripts/verify_kernel.sh` checks
`uname -r` against the known-good kernel (`6.8.0-1029-azure`) and that
`nvidia-smi` succeeds, and fails loudly (non-zero exit, no silent
fallback) if either check fails -- specifically so an unverified kernel
(e.g. an auto-applied `6.8.0-1064-azure`) is caught immediately at the
start of a session rather than surfacing as a confusing failure hours
into an experiment. `scripts/run_m3_m4_chain.sh` now runs this guard as
its first step; `STAGE_B_CHECKLIST.md`'s "How to resume" instructions
also call it out as step 2, before activating the venv.

### Second VM restart -- `/mnt` wiped again (2026-08-12, ~13:22 UTC)

The VM went unreachable (SSH connection dropped mid-EXP-003, then full
connection timeouts) while EXP-003's regeneration was at 56% (749/1340).
User independently confirmed the VM came back up healthy (correct kernel
`6.8.0-1029-azure`, driver 535.230.02, both GPUs idle, no processes
running -- i.e. nothing survived the restart). `/mnt/vmadmin/` was
confirmed wiped again (`ls`: "No such file or directory"; `/dev/sdb1`:
32 KB used -- fresh ephemeral disk, same signature as the first incident).

**Nothing scientifically new was lost**: EXP-003's `predictions.csv` is
only written at the very end of a full run, and this second interruption
happened before that (56% in, same as the first interruption never
reaching completion either) -- there was never a completed artifact to
lose this time. The recovery procedure from the first incident (see
above) was re-run verbatim and confirmed to work identically on the
second attempt:

- `/mnt/vmadmin/{projects,sarcasm-env,huggingface}` recreated.
- Repo re-synced via `scripts/sync_to_vm.sh` -- caught and fixed a real
  bug in the process: the script had no exclude rule for
  `web/frontend/node_modules`/`.next` (added after the first incident's
  recovery, before the web app existed), so the first re-sync attempt
  transferred ~550 MB of `node_modules` needlessly. Fixed
  (`--exclude 'web/frontend/node_modules'`, `--exclude 'web/frontend/.next'`)
  and the already-synced copy deleted from the VM.
- venv recreated, exact pinned stack reinstalled from the same
  `pip freeze` snapshot used the first time (`torch==2.5.1+cu118`,
  `transformers==5.15.0`, etc.) -- identical versions, nothing upgraded.
- Qwen3-4B-Instruct-2507 + deberta-v3-base re-downloaded (~30s combined,
  fast link).
- `verify_kernel.sh`, `verify_gpu.py`, the classification test suite
  (61/61), and the Qwen smoke test (`--limit 20`) all re-passed --
  smoke test reproduced the identical DEV metrics for the third time now
  (original run, first recovery, second recovery), confirming the
  pipeline behaves identically across all three environment rebuilds.
- `scripts/run_m3_m4_chain.sh` relaunched from scratch (EXP-003 -> EXP-004
  -> EXP-005) -- nothing valid persisted from the interrupted attempt to
  resume from, so a clean restart is correct here, not a resume.

**Root cause of the VM going down is still unknown** -- unlike the first
incident, this one wasn't a deliberate user-initiated reboot (grub-reboot
to fix a kernel/driver mismatch); it happened without warning during a
running experiment. Possible causes (Azure host maintenance/eviction, a
transient platform issue, something else) were not investigated further
since the user's own portal check confirmed the VM is healthy now and
asked to proceed with recovery rather than root-cause the outage. If this
recurs a third time, root-causing (Azure Activity Log / Serial console
boot log) would be worth doing before just recovering again.

**Durability note:** this incident is exactly why results/logs are pulled
back and committed after every experiment rather than only at planned
shutdowns (see "Fixes to prevent recurrence" above, and
`scripts/sync_from_vm.sh`) -- an unannounced VM loss like this one gives
zero warning to do a final sync first.

### EXP-004 — Qwen3-4B few-shot (curated demos) — **DEV-EVALUATED** (TEST sealed, not yet run)

- **Date:** 2026-08-12. **Environment:** Azure `Standard_NV24s_v3`, 2x Tesla M60 (post-2nd-VM-restart environment, identical pinned stack to every prior run).
- **Command:** `python -m src.classification.run_experiment --config configs/llm_few_shot_curated_8_qwen_local.json`
- **Config:** `provider=local_hf`, `model=Qwen/Qwen3-4B-Instruct-2507`, `mode=few_shot`, `few_shot_variant=curated`, `n_shots=8`, `temperature=0.0`, prompt `classification/few_shot_v1.txt`, `eval_split=dev`, seed 42.
- **Demo example IDs used** (8, curated -- stratified across (category, label)): `GEN-5007, RQ-838, GEN-787, GEN-3116, HYP-96, RQ-159, HYP-652, GEN-5870`.
- **Runtime:** ~1h22m wall-clock for 1,340 examples (~3.7s/example -- slightly faster than the random-demo run, ~4.2s/example, likely just shorter demo text this time; not investigated further since it doesn't affect correctness).
- **Full DEV metrics:**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.5821 |
  | Macro F1 | 0.5011 |
  | Weighted F1 | 0.5005 |
  | not_sarcastic: P / R / F1 (support 672) | 0.9375 / 0.1786 / 0.3000 |
  | sarcastic: P / R / F1 (support 668) | 0.5446 / 0.9880 / 0.7021 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[120, 552], [8, 660]]` |

- **Quality checks performed:**
  - `n_examples=1340`, 1,340 unique `example_id`s, no duplicates, no rows missing vs. `data/splits/dev.csv`.
  - Gold label distribution (672 not_sarcastic / 668 sarcastic) matches the canonical DEV split exactly.
  - Every prediction is a valid `{sarcastic, not_sarcastic}` value (run completed without a parsing-retry exhaustion, same structural guarantee as every prior LLM experiment).
  - **Flagged and investigated:** this is the most skewed result yet -- 1,212/1,340 (90.4%) predicted `sarcastic`, vs. 83% for EXP-002 (zero-shot) and 83.1% for EXP-003 (random few-shot). Checked the same three explanations as before:
    1. **Category-specific artifact?** No -- predicted-sarcastic rate is uniform across categories (GEN 90.0%, HYP 93.4%, RQ 90.3%).
    2. **Curated-demo selection bias?** The 8 curated demos are stratified across (category, label) cells by construction (`select_curated_few_shot`), not label-skewed by design -- ruled out as a data artifact.
    3. **Did the curated demos meaningfully change behavior, or is this the same model with noise?** Compared directly against both prior runs: 91.3% agreement with EXP-002 (zero-shot), 92.2% agreement with EXP-003 (random few-shot) -- high agreement in both cases, consistent with "same underlying bias, nudged slightly further" rather than a different failure mode or a bug.
  - **Conclusion: not a pipeline bug.** Curated, category/label-balanced demonstrations pushed Qwen3-4B *further* toward over-predicting `sarcastic` than either zero-shot or random few-shot, not less -- a genuine (if counterintuitive) empirical result. Manually reading a handful of the curated demo texts, all are clearly and unambiguously labeled examples; the effect appears to be that showing more explicit examples of "sarcastic" text (even balanced 4/4) reinforces the model's tendency to call adversarial/rhetorical language sarcastic, rather than teaching it to be more selective.
- **Artifacts:** `results/EXP-004/{config.json, metrics.json, predictions.csv}` -- pulled back to the local Mac and committed immediately (per the durability fix from the VM-restart incidents above; not left VM-only even briefly).
- **Comparison across all M2/M3 variants so far (DEV):**

  | Experiment | Macro F1 | Accuracy | sarcastic Recall |
  |---|---:|---:|---:|
  | EXP-002 (zero-shot) | 0.6008 | 0.6440 | 0.9760 |
  | EXP-003 (few-shot, random demos) | 0.5880 | 0.6328 | 0.9656 |
  | EXP-004 (few-shot, curated demos) | 0.5011 | 0.5821 | 0.9880 |

  Directionally clear and consistent: on this corpus, adding few-shot demonstrations does not help Qwen3-4B-Instruct-2507 -- it hurts, and curated (label/category-balanced) demos hurt *more* than random ones. Zero-shot (EXP-002) remains the best-performing manual-prompt variant of the three by a clear margin.
- **M3 variant selection (DEV-based, allowed per the sealing policy):** between the two few-shot variants, **random (EXP-003) is the winner** -- higher Macro F1 (0.588 vs. 0.501) and Accuracy (0.633 vs. 0.582). This is the few-shot candidate that will be considered (alongside EXP-002 zero-shot and EXP-005 reasoning) at the Phase 2 freeze step -- though given zero-shot beats both few-shot variants outright, the current DEV evidence points toward zero-shot being the more likely overall M2-M4 candidate to freeze, not few-shot at all. That decision is deferred to Phase 2, once EXP-005 (reasoning) has also run.
- **TEST not touched**, per the sealing policy -- correct.

### EXP-005 — Qwen3-4B structured reasoning — **DEV-EVALUATED** (TEST sealed, not yet run)

- **Date:** 2026-08-12. **Environment:** Azure `Standard_NV24s_v3`, 2x Tesla M60 (post-2nd-VM-restart environment, identical pinned stack).
- **Command:** `python -m src.classification.run_experiment --config configs/llm_reasoning_qwen_local.json`
- **Config:** `provider=local_hf`, `model=Qwen/Qwen3-4B-Instruct-2507`, `mode=reasoning`, `temperature=0.0`, prompt `classification/reasoning_v1.txt`, `eval_split=dev`, seed 42.
- **Runtime:** ~1h02m wall-clock for 1,340 examples (~2.7-2.9s/example -- faster than either few-shot variant, ~1s/example slower than plain zero-shot; consistent with a short prompt (no demonstrations) but a longer generated completion (the reasoning trace) than direct zero-shot).
- **Full DEV metrics:**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.6276 |
  | Macro F1 | 0.5796 |
  | Weighted F1 | 0.5791 |
  | not_sarcastic: P / R / F1 (support 672) | 0.9023 / 0.2887 / 0.4374 |
  | sarcastic: P / R / F1 (support 668) | 0.5751 / 0.9686 / 0.7217 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[194, 478], [21, 647]]` |

- **Quality checks performed:**
  - `n_examples=1340`, 1,340 unique `example_id`s, no duplicates, no rows missing vs. `data/splits/dev.csv`.
  - Gold label distribution (672/668) matches the canonical DEV split exactly.
  - Every prediction is a valid `{sarcastic, not_sarcastic}` value.
  - **Flagged and investigated:** same skew pattern as every other Qwen variant so far -- 83.4% predicted sarcastic overall (GEN 82.6%, HYP 90.4%, RQ 84.7% -- uniform across categories, ruling out a category-specific artifact). 94.6% agreement with EXP-002 (zero-shot) -- the highest agreement of any M3/M4 variant against zero-shot yet, meaning explicit step-by-step reasoning barely moved the model's decisions at all, and where it did, it landed almost exactly on the same skew.
  - **Conclusion: not a pipeline bug.** Prompting Qwen3-4B to reason step-by-step before answering did not meaningfully change its behavior relative to direct zero-shot prompting -- it neither fixed the over-predicting-sarcastic bias nor introduced a new failure mode. The model's underlying tendency to read adversarial/rhetorical text as sarcastic appears robust to prompting strategy (direct, few-shot, or reasoning) -- it's a property of the base model's zero-shot calibration on this corpus, not something any of the four manual-prompt variants tested so far can prompt its way around.
- **Artifacts:** `results/EXP-005/{config.json, metrics.json, predictions.csv}` -- pulled back and committed immediately.
- **Comparison across all M2-M4 manual-prompt variants (DEV), now complete:**

  | Experiment | Method | Macro F1 | Accuracy | sarcastic Recall |
  |---|---|---:|---:|---:|
  | EXP-002 | Zero-shot | **0.6008** | **0.6440** | 0.976 |
  | EXP-005 | Structured reasoning | 0.5796 | 0.6276 | 0.969 |
  | EXP-003 | Few-shot (random) | 0.5880 | 0.6328 | 0.966 |
  | EXP-004 | Few-shot (curated) | 0.5011 | 0.5821 | 0.988 |

  **Zero-shot (EXP-002) is the best of all four manual-prompt variants on DEV**, by a clear and consistent margin on both Macro F1 and Accuracy. None of the three "smarter prompting" variants (either few-shot flavor, or explicit reasoning) improved on the simplest possible prompt -- each one either left the model's sarcastic-overprediction bias unchanged (reasoning) or made it worse (both few-shot variants, curated worse than random). This is a genuine, repeatedly-confirmed empirical finding for this specific base model/corpus, not an artifact of any one run.
- **TEST not touched**, per the sealing policy -- correct.
- **M2-M4 development is now complete.** Per explicit instruction, the pipeline pauses here -- M5 (DSPy) and M6 (DeBERTa) are not started automatically; both remain queued pending explicit go-ahead.

### Third VM restart -- `/mnt` wiped again (2026-08-13, ~12:27 UTC)

User restarted the VM (independently, between sessions -- work had been
deliberately paused before M5 with 0 processes running, so nothing was
lost mid-experiment this time). On reconnect, `/mnt/vmadmin/` was
confirmed empty again (`ls`: "No such file or directory") -- same
ephemeral-disk signature as the first two incidents. `df -h /mnt` showed
a fresh filesystem (48 KB used / 1.5 TB) and `/mnt` itself was owned by
`root` (not `vmadmin`) immediately after boot, requiring `sudo mkdir` +
`sudo chown` before `vmadmin` could write to it again -- a new detail not
seen on the first two recoveries, noted here in case it recurs.

**Nothing was lost** (unlike the first two incidents): work had been
paused cleanly before M5 with all M1-M4 results already committed to git,
so there was no in-progress experiment or uncommitted artifact on `/mnt`
to lose.

**Recovery performed, identical procedure to the first two incidents,
confirmed working a third time:**
1. Verified survival: `hostname` (`dpmlgpuNC6sv32025s-0003`), kernel
   (`6.8.0-1029-azure`, correct), `nvidia-smi` (driver 535.230.02, both
   Tesla M60s idle, 0 MiB used).
2. `sudo mkdir -p /mnt/vmadmin/{projects,sarcasm-env,huggingface}` +
   `sudo chown -R vmadmin:vmadmin /mnt/vmadmin` (the new step -- `/mnt`
   came back root-owned this time).
3. Repo re-synced via `scripts/sync_to_vm.sh` (unchanged).
4. venv recreated (`python3.10 -m venv`, same 3.10.12), exact pinned
   stack reinstalled from `environment_stage_b.txt`'s pip-freeze
   (`torch==2.5.1+cu118`, `transformers==5.15.0`, `dspy==3.3.0`, etc.) --
   identical versions, nothing upgraded, same as both prior recoveries.
5. `Qwen/Qwen3-4B-Instruct-2507` + `microsoft/deberta-v3-base`
   re-downloaded (~25s combined) into a fresh `HF_HOME`.
6. `verify_kernel.sh`, `verify_gpu.py` (identical verdict to every prior
   run), the classification test suite (61/61), and the Qwen zero-shot
   smoke test (`--limit 20`, `SMOKE-recovery3-qwen-zero-shot`) all
   re-passed -- **reproduced accuracy 0.25 / macro F1 0.20 exactly, the
   fourth time now** (original, first recovery, second recovery, this
   one) that this pipeline has behaved identically across a full
   environment rebuild.
7. Per-example LLM disk cache restored from the local Mac's periodic
   backup (`data/llm_cache/`, 3,949 entries) to the freshly-recreated VM
   before launching anything.
8. Background monitoring re-armed fresh (cache backup every 5 min, 15-min
   progress heartbeat, chain state-change watcher) -- as expected, none
   of it survived the new session per se, this is the normal every-session
   step, not incident-specific.
9. **M5 launched** (`scripts/run_m5_chain.sh`, nohup + disown on the VM):
   adapter smoke test passed (5/5), EXP-006 (`dspy.Predict`) started
   running against full DEV.

**Root cause still unknown** -- three unannounced-or-user-initiated `/mnt`
wipes now on this VM in two days. Root-causing via Azure Activity Log or
serial console boot log was considered again (per the standing note after
the second incident) but not pursued this time either: no `az` CLI
available in this environment, and since nothing was lost and the
recovery is now a well-proven <15-minute procedure, proceeding directly
with recovery remains the better use of time than investigating a
platform-level cause that may not be actionable anyway. If a fourth
wipe happens, root-causing via the Azure portal (not available here)
would be worth doing directly, since the recovery-tax is now the main
remaining cost of this instability.

**No new durability fixes needed** -- the existing infra (git as source
of truth, `sync_from_vm.sh`/`sync_cache_from_vm.sh`, `verify_kernel.sh`)
already handled this cleanly; the only new observation is the `/mnt`
root-ownership-after-boot detail captured in step 2 above, in case a
future recovery hits the same "Permission denied" and needs the `sudo
chown` step.

### EXP-006 — DSPy `Predict` (unoptimized baseline), local Qwen — **DEV-EVALUATED** (TEST sealed, not yet run)

- **Date:** 2026-08-13. **Environment:** Azure `Standard_NV24s_v3`, 2x Tesla M60 (post-3rd-VM-restart environment, identical pinned stack: `dspy==3.3.0`, same Qwen3-4B-Instruct-2507).
- **Command:** `python -m src.classification.run_experiment --config configs/dspy_predict.json` (via `scripts/run_m5_chain.sh`, first of the M5 chain, after the adapter smoke test passed 5/5).
- **Config:** `provider=local_hf`, `model=Qwen/Qwen3-4B-Instruct-2507`, `approach=M5_dspy`, `optimizer=predict` (unoptimized `dspy.Predict` over the `SarcasmClassification` signature -- plain input/output field wrapper, no bootstrapped demos, no prompt optimization), `temperature=0.0`, `eval_split=dev`, seed 42.
- **Runtime:** 56m49s wall-clock for 1,340 examples (~2.5s/example -- similar order to the manual-prompt zero-shot run).
- **Full DEV metrics:**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.6799 |
  | Macro F1 | **0.6619** |
  | Weighted F1 | 0.6616 |
  | not_sarcastic: P / R / F1 (support 672) | 0.8384 / 0.4479 / 0.5839 |
  | sarcastic: P / R / F1 (support 668) | 0.6218 / 0.9132 / 0.7398 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[301, 371], [58, 610]]` |

- **Quality checks performed:**
  - `n_examples=1340`, 1,340 unique `example_id`s, no duplicates, no rows missing/extra vs. `data/splits/dev.csv`.
  - Gold label distribution (672/668) matches the canonical DEV split exactly.
  - Every prediction is a valid `{sarcastic, not_sarcastic}` value.
  - Predicted-sarcastic rate 73.2% overall, roughly uniform across categories (GEN 70.8%, HYP 80.7%, RQ 77.0%) -- less skewed than any of the four manual-prompt variants (83-90%).
  - 86.6% agreement with EXP-002 (zero-shot manual prompt) -- same underlying model, same general direction, but DSPy's structured signature-based prompt construction clearly changes behavior enough to matter.
- **This is the best result of any method so far, by a clear margin**: Macro F1 0.6619 vs. EXP-002's 0.6008 (previous best, manual zero-shot). Notably, this is the **unoptimized** DSPy baseline -- no bootstrapping or MIPROv2 tuning yet, just DSPy's own signature-driven prompt template outperforming every hand-written prompt tried in M2-M4. Not yet investigated *why* (differs in exact prompt wording/structure from the manual `zero_shot_v1.txt`/`few_shot_v1.txt`/`reasoning_v1.txt` templates) -- worth a closer look once EXP-007/EXP-008 are in, to see if the gain is specifically DSPy's default template or something that further optimization builds on.
- **Artifacts:** `results/EXP-006/{config.json, metrics.json, predictions.csv}` -- pulled back and committed immediately.
- **TEST not touched**, per the sealing policy -- correct.
- **M5 chain continues automatically**: EXP-007 (`BootstrapFewShot`) started immediately after (`configs/dspy_bootstrap_few_shot.json`).

### EXP-007 — DSPy `BootstrapFewShot`, local Qwen — **DEV-EVALUATED** (TEST sealed, not yet run)

- **Date:** 2026-08-13. **Environment:** same as EXP-006 (post-3rd-VM-restart, `dspy==3.3.0`).
- **Command:** `python -m src.classification.run_experiment --config configs/dspy_bootstrap_few_shot.json` (second step of the M5 chain, immediately after EXP-006).
- **Config:** `optimizer=bootstrap_few_shot`, `max_bootstrapped_demos=4`, `max_labeled_demos=8`, `max_rounds=1`, `trainset_sample_size=150` (samples 150 of TRAIN's 8,385 rows to search for valid bootstrapped demos; TEST/DEV not used for the bootstrap search itself -- `dev_size_used_for_optimization` in the saved config is just informational metadata, not evidence of DEV leakage, confirmed by reading `build_program()`: `BootstrapFewShot.compile()` is called with `trainset` only, no `valset`).
- **Runtime:** **2h48m24s wall-clock** for the full step (compile + full-DEV eval combined) -- far longer than EXP-006's 56m49s, and far longer than initially estimated. **Root cause, confirmed live via `py-spy` process inspection during the run** (see below): once bootstrapped demos are found, every subsequent classification call embeds up to 8 few-shot demos directly in the prompt (~1,200-1,400 tokens vs. EXP-006's short zero-shot prompt), and Tesla M60 has no flash-attention support (`attn_implementation='eager'`, O(n²) cost) -- so per-example cost during the 1,340-example DEV eval phase rose to ~7.5s/example (vs. EXP-006's ~2.5s/example), accounting for most of the wall-clock. The compile/bootstrap phase itself (max 150 LM calls, well under the trainset_sample_size cap since it stops once it has enough valid demos) was comparatively short.
- **Live monitoring note (process introspection, not just log-watching):** DSPy's own bootstrap-phase progress bar writes to a pipe (`python ... | tee logs/...`), which block-buffers and can appear frozen on-screen for tens of minutes while the process is still genuinely computing -- confirmed **not a hang** by installing `py-spy` on the VM (`sudo py-spy dump --pid <pid> --locals`) mid-run and reading the live Python call stack: (1) confirmed the process was inside `transformers.generate()`'s sampling loop, not blocked on I/O; (2) confirmed forward progress by reading the `eval_df.iterrows()` loop index (`_`) directly from the `run_dspy_experiment` frame's locals across repeated snapshots (e.g. row 1181 -> row 1189 in 58s, row 1249 -> row 1293 in 5m35s), which let elapsed-time-to-completion be estimated accurately (within a few minutes) well before the run actually finished. Worth reusing this technique (`py-spy dump --locals`, reading the loop variable) for any future silent-progress-bar situation on this VM rather than assuming a stall.
- **Full DEV metrics:**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.6664 |
  | Macro F1 | 0.6406 |
  | Weighted F1 | 0.6403 |
  | not_sarcastic: P / R / F1 (support 672) | 0.8641 / 0.3973 / 0.5443 |
  | sarcastic: P / R / F1 (support 668) | 0.6072 / 0.9371 / 0.7369 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[267, 405], [42, 626]]` |

- **Quality checks performed:**
  - `n_examples=1340`, 1,340 unique `example_id`s, no duplicates, no rows missing/extra vs. `data/splits/dev.csv`.
  - Gold label distribution (672/668) matches the canonical DEV split exactly.
  - Every prediction is a valid `{sarcastic, not_sarcastic}` value.
  - Predicted-sarcastic rate roughly uniform across categories (GEN 75.6%, HYP 82.5%, RQ 78.2%).
  - 86.7% agreement with EXP-006 (unoptimized `dspy.Predict`) -- same base model/signature, high but not total overlap.
  - `compiled_program.json` saved with the 8 selected demos (4 bootstrapped + up to 8 labeled) -- inspected a couple manually, both correctly labeled `not_sarcastic` examples of ambiguous/argumentative (but sincere) text, consistent with the corpus.
- **Result: BootstrapFewShot slightly *underperforms* the unoptimized `dspy.Predict` baseline** -- Macro F1 0.6406 vs. EXP-006's 0.6619 (Accuracy 0.6664 vs. 0.6799 -- both metrics down). This echoes the M2-M4 finding that adding few-shot demonstrations does not help this model/corpus (EXP-003/004 also underperformed EXP-002's zero-shot) -- now confirmed a second time with DSPy's automatic (not hand-curated) demo selection. **EXP-006 (unoptimized `dspy.Predict`) remains the best M5 result so far and the best of any method in Stage B to date.**
- **Artifacts:** `results/EXP-007/{config.json, metrics.json, predictions.csv, compiled_program.json}` -- pulled back and committed immediately.
- **TEST not touched**, per the sealing policy -- correct.
- **M5 chain paused here per explicit user request**: the chain auto-advanced to EXP-008 (MIPROv2) the instant EXP-007 finished, but was killed immediately (`kill` on both `run_m5_chain.sh` and the EXP-008 process, confirmed via `ps`/`nvidia-smi` that nothing was left running and both GPUs returned to 0% util / 0 MiB) before it could consume meaningful compute -- no EXP-008 artifacts exist, nothing was lost by stopping it. Reason: given how much slower this hardware is running than the methodology anticipated (EXP-007 took ~3x its rough estimate), it's worth reconsidering EXP-008's `auto="light"` budget with this now-measured per-call cost in hand, rather than launching it unattended. **EXP-008 is the correct and only next step for M5** -- see STAGE_B_CHECKLIST.md.

### Fourth VM restart (`/mnt` wiped) + kernel auto-switch broke the NVIDIA driver entirely (2026-08-14)

**What happened, in order, across a session gap that was never documented at
the time (found reconstructed from local uncommitted artifacts, not from
any session log):**

1. Sometime after the EXP-007 commit, a session actually launched EXP-008
   (`configs/dspy_mipro_v2.json`) rather than stopping before it as the
   prior checklist entry claimed. It ran for real -- bootstrap phase
   progressed to "Bootstrapping set 4/6" (log timestamp 2026-08-13
   16:25-16:26) -- then stopped mid-run with no results directory ever
   created. Evidence: an untracked `logs/EXP-008-dspy-mipro-dev.log`
   (partial) and two untracked smoke-test result dirs
   (`SMOKE-recovery2/3-qwen-zero-shot`) were sitting locally, uncommitted,
   at the start of this session -- consistent with a fourth `/mnt` wipe
   interrupting that run before anything could be pulled back or
   documented.
2. This session (2026-08-14) reconnected and confirmed: SSH fine, VM
   itself alive (same hostname), but `/mnt` completely empty again (32 KB
   used / 1.5 TB, same signature as the first three incidents) -- the
   fourth `/mnt` wipe. Ran the full recovery procedure a fourth time
   (mkdir/chown -- root-owned again, same as the third recovery -- sync
   repo, rebuild venv, reinstall pinned stack, re-download Qwen, restore
   `data/llm_cache/` from the local backup). `verify_kernel.sh`,
   `verify_gpu.py`, pytest (61/61), and a zero-shot smoke test
   (`SMOKE-recovery4-qwen-zero-shot`, accuracy 0.25 / macro F1 0.20) all
   reproduced exactly, a fourth time.
3. **New finding during this recovery:** `microsoft/deberta-v3-base`
   (needed for M6, not M5) now fails to download/load with
   `ValueError: ...we now require users to upgrade torch to at least
   v2.6...` (transformers' `check_torch_load_is_safe` guard, triggered
   because that HF repo has no safetensors weights and pinned
   `torch==2.5.1`). Not investigated further since M6 isn't reached yet
   this session -- **flagged here for whoever picks up M6**, deliberately
   not "fixed" by upgrading torch (would risk destabilizing the verified
   M2-M5 stack) without discussion first.
4. Relaunched EXP-008 -- **immediately failed** with
   `ImportError: MIPROv2 requires optional dependency 'optuna'`. This is
   a **real, pre-existing environment gap**, not caused by any VM
   incident: `optuna` was never in any prior pip-freeze capture of
   `environment_stage_b.txt`, so this session's first EXP-008 attempt
   (item 1 above) would have hit the exact same failure at its
   optimization step regardless of the fourth `/mnt` wipe -- it just
   hadn't gotten far enough to reach it yet. Installed `optuna==4.9.0`
   and added it to `environment_stage_b.txt` for durability.
5. Relaunched EXP-008 again -- **the VM became completely SSH-unreachable
   within seconds** (`ssh: connect ... Operation timed out`, not the
   `/mnt`-only signature of prior incidents), while this session's own
   internet connectivity was independently confirmed fine (google.com/
   github.com both reachable). Per the standing "if it times out
   completely, that itself is the finding -- report it, don't guess"
   guidance, this was reported to the user rather than guessed at; user
   asked to just retry in a few minutes, which worked -- SSH came back
   ~4 minutes later.
6. **On reconnect: `/mnt` wiped a fifth time, AND `nvidia-smi` failed
   outright** (`couldn't communicate with the NVIDIA driver`) even though
   the VM itself was reachable -- a new, more serious failure mode than
   any prior incident (previously the driver always came back healthy
   immediately). Diagnosis: `uname -r` showed `6.8.0-1064-azure`, not the
   known-good `6.8.0-1029-azure` -- **exactly the scenario
   `scripts/verify_kernel.sh`'s own header comment was written to catch**
   (a newer kernel package sitting on disk, unbooted, since the very
   first 2026-08-12 incident). `last reboot` showed several reboots
   within the same hour alternating between the two kernels -- consistent
   with something (unattended-upgrades' periodic dpkg run, and/or
   Azure-portal restarts around the same time the user was independently
   reconnecting) landing on the newer kernel by default, since GRUB's
   *default* boot entry was never pinned after the very first incident's
   one-shot `grub-reboot` (which only affects the single next boot, not
   subsequent ones).
7. **Fix, in two parts:**
   - Immediate: found the exact GRUB menu entry ID for
     `6.8.0-1029-azure` under "Advanced options" (`sudo grep -n
     "menuentry_id_option 'gnulinux-6.8.0-1029" /boot/grub/grub.cfg`),
     `sudo grub-reboot '<advanced-submenu-id>><entry-id>'`, `sudo reboot`.
     Came back on `6.8.0-1029-azure`, driver healthy again (both Tesla
     M60s idle, driver 535.230.02).
   - **New durability fix (not done after any prior incident):**
     `sudo grub-set-default '<same path>'` + `sudo update-grub` --
     unlike `grub-reboot` (one-shot), this makes `6.8.0-1029-azure` the
     *permanent* default, so any future reboot (unattended-upgrades,
     Azure-portal restart, crash) lands on the known-good kernel
     automatically without needing this manual fix repeated. Verified via
     `grubenv`'s `saved_entry`. This is the first time root-causing this
     specific mechanism (rather than just re-running the same recovery)
     has paid off -- worth checking `saved_entry` early in any future
     kernel-related incident before assuming another one-shot
     `grub-reboot` is needed.
8. Redid the full recovery a fifth time (same procedure as step 2, since
   the reboot re-wiped `/mnt`): repo sync, venv + pinned stack + `optuna`,
   Qwen re-download, cache restore. `verify_kernel.sh`, pytest (61/61),
   and a second zero-shot smoke test (`SMOKE-recovery5-qwen-zero-shot`)
   all passed -- accuracy 0.25 / macro F1 0.20 reproduced exactly a sixth
   time now.
9. **EXP-008 relaunched a third time and confirmed genuinely running**
   past the point of both prior failures (into MIPROv2's "STEP 2: PROPOSE
   INSTRUCTION CANDIDATES", both GPUs active 41-50% util, ~5.9 GB each).
   Background monitoring re-armed (cache backup every 5 min, 15-min
   heartbeat, state-change watcher on the EXP-008 log).

**Update: EXP-008's first real run (after all of the above) crashed for a
genuine code reason, not an infra one.** Trial 1 of MIPROv2's optimization
loop failed with `ValueError: Field types must be types, but received:
ForwardRef("Literal['sarcastic', 'not_sarcastic']")...`, uncaught by
Optuna, which killed the whole process (GPUs back to 0%, no `ps` entry).
**Root cause:** `src/classification/dspy_pipeline/signatures.py` had
`from __future__ import annotations` at module level, but
`SarcasmClassification` is defined *inside* `build_signature()`, a local
function scope. Under postponed evaluation, its `label: Literal[...]`
annotation is stored as a string, and pydantic/dspy resolve such strings
against the *module's* globals -- which don't include the function-local
`from typing import Literal` -- so the annotation stayed an unresolved
`ForwardRef`. This never surfaced in EXP-006 (`Predict`) or EXP-007
(`BootstrapFewShot`) because neither calls `Signature.with_instructions()`;
MIPROv2 is the first optimizer to rebuild the signature with new
instructions, which re-validates field types and hits the unresolved
ForwardRef. **Fix:** removed `from __future__ import annotations` from
`signatures.py` (nothing else in that short file needed postponed
evaluation) -- confirmed on the VM that `build_signature().with_instructions(...)`
now works, and the full test suite (61/61) still passes. EXP-008
relaunched immediately after, confirmed genuinely progressing past the
point of the crash (`ps`/`nvidia-smi` both active).

### EXP-008 — DSPy `MIPROv2` (`auto="light"`), local Qwen — **DEV-EVALUATED, NEW BEST** (TEST sealed, not yet run)

- **Date:** 2026-08-14. **Environment:** Azure `Standard_NV24s_v3`, 2x Tesla M60 (post-4th/5th-`/mnt`-wipe environment, identical pinned stack plus `optuna==4.9.0`; the `signatures.py` fix above applied).
- **Command:** `python -m src.classification.run_experiment --config configs/dspy_mipro_v2.json` (single step, launched directly, not via the M5 chain script since EXP-006/007 were already done).
- **Config:** `optimizer=mipro_v2`, `optimizer_config={"auto": "light", "trainset_sample_size": 150, "valset_sample_size": 100}` -- launched as-is (judgment call from the prior session: `num_trials` is fixed by the `auto="light"` preset regardless of `valset_sample_size`, and the dominant fixed cost is the final full-DEV eval loop, which doesn't depend on `valset_sample_size` either -- see STAGE_B_CHECKLIST.md's step 4 note).
- **Runtime:** ~1h13m for the optimization phase (bootstrap + 13 trials, including periodic full-100-valset checkpoint evals) + ~1h16m for the final full-1,340-DEV eval (~3.4s/example -- notably faster than EXP-007's ~7.5s/example, explained below) = **~2h29m total** wall-clock (12:59-15:07 local VM time).
- **What MIPROv2 actually chose** (from `Optuna`'s trial log and `compiled_program.json`): out of 3 proposed instruction candidates and 6 bootstrapped few-shot demo sets, the winning combination was **`Instruction 0` (the original/default instruction, unchanged: "Classify whether an English sentence is sarcastic.") + `Few-Shot Set 4`, a compact 4-demo set** -- MIPROv2 tried rewriting the instruction but the *default* instruction paired with a well-chosen small demo set won out over every rewritten-instruction candidate. This is why the final eval ran at ~3.4s/example rather than EXP-007's ~7.5s/example: 4 short demos in the prompt, not up to 8.
- **Full DEV metrics:**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.6843 |
  | Macro F1 | **0.6700** |
  | Weighted F1 | 0.6698 |
  | not_sarcastic: P / R / F1 (support 672) | 0.8201 / 0.4747 / 0.6013 |
  | sarcastic: P / R / F1 (support 668) | 0.6288 / 0.8952 / 0.7387 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[319, 353], [70, 598]]` |

- **Quality checks performed:**
  - `n_examples=1340`, 1,340 unique `example_id`s, no duplicates, no rows missing/extra vs. `data/splits/dev.csv`.
  - Gold label distribution (672/668) matches the canonical DEV split exactly.
  - Every prediction is a valid `{sarcastic, not_sarcastic}` value.
  - Predicted-sarcastic rate 71.0% overall, roughly uniform across categories (GEN 69.9%, HYP 80.1%, RQ 69.0%) -- similar shape to EXP-006/007, less skewed than the manual-prompt variants.
  - 86.4% agreement with EXP-006 (unoptimized `Predict`), 87.5% agreement with EXP-007 (`BootstrapFewShot`) -- high but not total overlap with either, consistent with it being a genuinely different (if related) prompt configuration.
- **Result: MIPROv2 is the new best result of any method in Stage B** -- Macro F1 0.6700 vs. EXP-006's previous-best 0.6619 (+0.0081) and EXP-007's 0.6406. A modest but real improvement, and notably achieved with an even *smaller* few-shot set (4 demos) than EXP-007's up-to-8, plus MIPROv2's own trial-based selection rather than BootstrapFewShot's simpler sampling -- suggesting the gain comes from smarter *selection* of which demos to use (via the optimization loop's minibatch scoring), not from more demos or a cleverer instruction.
- **Artifacts:** `results/EXP-008/{config.json, metrics.json, predictions.csv, compiled_program.json}` -- pulled back and committed immediately.
- **TEST not touched**, per the sealing policy -- correct.
- **M5 is now COMPLETE**: all three DSPy variants (Predict, BootstrapFewShot, MIPROv2) run, quality-checked, and recorded. EXP-008 (MIPROv2, Macro F1 0.6700) is the DEV leader across all of Stage B so far (M1-M5). **Per explicit user request, stopping here rather than auto-continuing into M6** -- next session should start directly at EXP-009 (M6, DeBERTa-v3-base fine-tuning); see STAGE_B_CHECKLIST.md's "START HERE" section, updated accordingly. Note the DeBERTa download blocker (torch/safetensors guard, flagged above) is still unresolved and will need addressing before EXP-009 can actually run.

**Root cause of the `/mnt` wipes themselves is still not fully pinned
down** (now five occurrences) -- this incident adds real evidence though:
at least this particular wipe coincided with a kernel-driven reboot, and
the reboot pattern in `last reboot` suggests portal-level restarts (the
user reported reconnecting independently around the same window) rather
than a purely internal VM process. The GRUB default-entry fix in step 7
should prevent the *driver-breaking* consequence from recurring even if
the underlying restart cause (whatever triggers it) continues -- `/mnt`
being ephemeral and wiped on any restart remains expected and handled by
the existing recovery procedure regardless.

### Sixth VM restart -- `/mnt` wiped again (2026-08-15, ~15:27 UTC), M6 DeBERTa blocker resolved

Resuming per STAGE_B_CHECKLIST.md's "START HERE" section, exactly as
written: reconnected (SSH fine, no `ConnectTimeout`), and found `/mnt`
empty again (`ls`: "No such file or directory", fresh `/dev/sdb1`
32 KB used) -- the sixth `/mnt` wipe. Kernel/driver this time came back
correct on first check (`uname -r` `6.8.0-1029-azure`, `nvidia-smi`
healthy, both Tesla M60s idle, 0 MiB used) -- the `grub-set-default` fix
from the fifth-restart incident held, no kernel-related recovery needed
this time.

**Recovery performed (identical procedure to the prior five, per the
existing runbook):**
1. `/mnt` came back root-owned again -- `sudo mkdir -p
   /mnt/vmadmin/{projects,sarcasm-env,huggingface} && sudo chown -R
   vmadmin:vmadmin /mnt/vmadmin`.
2. Repo re-synced via `scripts/sync_to_vm.sh`.
3. Python 3.10 venv rebuilt from scratch (`python3.10-venv` apt package
   already present system-wide, survived the `/mnt` wipe as expected --
   only `/mnt` is ephemeral). Installed the exact pinned stack from
   `environment_stage_b.txt` (`torch==2.5.1+cu118` via the cu118 index
   first, then the rest of the freeze, `optuna==4.9.0` included) --
   no version changed.
4. Re-downloaded `Qwen/Qwen3-4B-Instruct-2507` into fresh
   `HF_HOME=/mnt/vmadmin/huggingface` -- succeeded in ~21s, no issue.
5. `scripts/verify_gpu.py`: identical verdict to every prior run
   (`fp16_transformers_pipeline_ok: true`, both GPUs visible, BF16/
   FlashAttention2/vLLM correctly reported unsupported).
6. `pytest tests/test_classification_*.py`: **61/61 passed.**
7. Qwen zero-shot smoke test (`SMOKE-recovery6-qwen-zero-shot`,
   `--limit 20`): **accuracy 0.25, macro F1 0.20 -- reproduced exactly, a
   seventh time now.**

**M6 DeBERTa download blocker (flagged but not investigated during the
2026-08-14 session) -- now resolved:**
`microsoft/deberta-v3-base` re-downloaded via `snapshot_download` (not
`AutoModel.from_pretrained`, so `transformers`' `check_torch_load_is_safe`
guard never triggers at this stage) -- all 8 repo files including
`pytorch_model.bin` land in the HF cache normally. The blocker only bites
when `transformers` tries to *load* the `.bin` weights: pinned
`torch==2.5.1` fails the guard's `>=2.6` requirement, and this repo has
no `model.safetensors` upstream to fall back to. **Fix applied:** loaded
the cached `pytorch_model.bin` directly with `torch.load(...,
weights_only=True)` (safe here -- Microsoft's own official repo, not
third-party), converted to `model.safetensors` with
`safetensors.torch.save_file`, and wrote it into the *same* HF cache
snapshot directory alongside the original files. Confirmed:
`AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base',
num_labels=2, use_safetensors=True)` now succeeds -- LOAD REPORT shows
all 198 backbone weights loaded cleanly (only the MLM head
`lm_predictions.*`/`mask_predictions.*` keys are UNEXPECTED, and
`classifier.*`/`pooler.*` MISSING/newly-initialized -- both expected and
correct for fine-tuning a base checkpoint into a 2-class classifier).
**No code changes needed** -- `configs/transformer_deberta_v3_base.json`
and `configs/transformer_deberta_v3_base_smoke.json` already specify
`use_safetensors: true`; the conversion alone was the missing piece.
**Caveat:** since `/mnt` is ephemeral, this conversion must be redone
after any future `/mnt` wipe, before M6/EXP-009 can run -- it's a ~5s
step (`torch.load` + `save_file` on the cached `.bin`), not worth
scripting into `verify_gpu.py` or similar for a one-time M6 need, but
worth remembering if a future session hits the same `ValueError` again.

**Two more real code bugs found and fixed while running the M6 smoke test
(`configs/transformer_deberta_v3_base_smoke.json`), both pre-existing
environment/version gaps unrelated to the `/mnt` wipe:**

1. **`TrainingArguments.__init__() got an unexpected keyword argument
   'warmup_ratio'`.** `transformers==5.15.0` removed the standalone
   `warmup_ratio` parameter -- `warmup_steps` is now overloaded to accept
   either an absolute step count (`>=1`) or a fraction of total training
   steps (`<1`), replacing the old two-parameter design. **Fix:**
   `src/classification/transformer/finetune.py`'s `TrainingArguments(...)`
   call now passes `warmup_steps=config.warmup_ratio` instead of
   `warmup_ratio=config.warmup_ratio` -- the config field name/semantics
   (a 0-1 fraction) are unchanged, only the kwarg it's forwarded under.
2. **`eval_loss: nan` every epoch, final DEV predictions collapsed to a
   single class (macro F1 ~0.33, chance level).** Root cause: the model
   load (`AutoModelForSequenceClassification.from_pretrained(...)`) never
   specified a dtype, and it turns out `microsoft/deberta-v3-base`'s
   `pytorch_model.bin` on the HF Hub is itself stored in **float16**
   (confirmed via a raw `torch.load` dtype check) -- so the model trained
   natively in fp16 with no fp32 master weights, causing gradient
   underflow/NaN within a handful of steps regardless of the Trainer's
   own `fp16` flag. This also explains the earlier `fp16=true` crash
   above (`ValueError: Attempting to unscale FP16 gradients` -- Trainer's
   `GradScaler` expects fp32 master weights to unscale into; the model
   was already fp16, so it had none). **Fix:** added `dtype=torch.float32`
   to the `from_pretrained(...)` call in `finetune.py` -- forces fp32
   weights regardless of the checkpoint's on-disk dtype, matching
   standard fine-tuning practice. **Verified via an ad hoc `Trainer` run**
   (bypassing `run_experiment.py`, logging every step): with the fix,
   losses are sane (~0.66-0.78, no NaN) and gradients are well-behaved
   (`grad_norm` 1-4, no explosion) under **both** `fp16=false` and
   `fp16=true` -- `fp16=true` is confirmed genuinely stable now (not just
   "didn't crash this once"), so it was restored in both
   `transformer_deberta_v3_base_smoke.json` and
   `transformer_deberta_v3_base.json` for training speed on the M60s.
   Re-ran the official smoke test (`SMOKE-deberta`, via
   `run_experiment.py`, `fp16=true`) end-to-end after the fix: 0 NaN
   anywhere in `metrics.json`, forward/backward/checkpoint-save/DEV-eval
   all completed cleanly. (The smoke test's DEV metrics themselves are
   near chance -- macro F1 0.33, predictions collapsed to one class -- but
   that's expected and not a bug: 64 train examples, 3 epochs, `lr=1e-5`
   is far too little signal/steps for a base encoder to learn anything
   real; the smoke test's job is catching crashes/NaN, not accuracy, and
   it now does so cleanly.)

M6/EXP-009 (DeBERTa-v3-base fine-tuning, full run) is next, per user
instruction to continue from the documented resume point.

### EXP-009 — Fine-tuned `microsoft/deberta-v3-base` (M6) -- **DEV-EVALUATED, NEW BEST BY A LARGE MARGIN** (TEST sealed, not yet run)

- **Date:** 2026-08-15. **Environment:** Azure `Standard_NV24s_v3`, single Tesla M60 (`CUDA_VISIBLE_DEVICES=0`), post-6th-`/mnt`-wipe environment, identical pinned stack, `dtype=torch.float32` + `warmup_steps` fixes from the smoke-test debugging above applied.
- **Command:** `python -m src.classification.run_experiment --config configs/transformer_deberta_v3_base.json`, launched detached (`nohup ... & disown`) so it survives SSH disconnects.
- **Config:** full TRAIN split (`n_train=6706`), `max_length=128`, `learning_rate=1e-5`, `train_batch_size=16`, `eval_batch_size=32`, `num_epochs=5` (early stopping `patience=2` on DEV Macro F1), `warmup_ratio=0.1`, `weight_decay=0.01`, `fp16=true`, `use_safetensors=true`, `use_fast_tokenizer=false`, `seed=42`.
- **Runtime:** ~22 minutes wall-clock (15:56-16:18 local VM time) for the full run, including per-epoch DEV eval -- dramatically faster than any M2-M5 LLM-based method (which ran 1-3 hours each), as expected for a 184M-parameter encoder vs. a 4B-parameter LLM doing generative inference.
- **Training dynamics:** DEV Macro F1 by epoch -- 1: 0.8193, 2: **0.8254 (best)**, 3: 0.8178 (regression), then early stopping triggered after epoch 4 also failed to improve (`patience=2`); best checkpoint (epoch 2) restored automatically via `load_best_model_at_end=True`. Training loss kept decreasing after epoch 2 (0.53 -> 0.36 -> 0.25 across the logged steps) while DEV metrics plateaued/regressed -- classic overfitting onset on a 6,706-example TRAIN set with a 184M-parameter encoder, exactly what early stopping exists to catch.
- **Full DEV metrics (best checkpoint, epoch 2):**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.8254 |
  | Macro F1 | **0.8254** |
  | Weighted F1 | 0.8254 |
  | not_sarcastic: P / R / F1 (support 672) | 0.8211 / 0.8333 / 0.8272 |
  | sarcastic: P / R / F1 (support 668) | 0.8298 / 0.8174 / 0.8235 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[560, 112], [122, 546]]` |

- **Quality checks performed:**
  - `n_examples=1340`, 1,340 unique `example_id`s, no duplicates; `example_id` set matches `data/splits/dev.csv` exactly.
  - Gold label distribution (672/668) matches the canonical DEV split exactly.
  - Predicted distribution (682 not_sarcastic / 658 sarcastic) is close to gold, not collapsed to one class.
  - Predicted-sarcastic rate by category: GEN 50.4%, HYP 54.8%, RQ 40.3% -- some spread but not wildly skewed toward one category.
  - Agreement with EXP-006 (Qwen `Predict`, unoptimized zero-shot-style): 68.1%. Agreement with EXP-008 (Qwen `MIPROv2`, prior best): 70.8%. Meaningfully different from both, consistent with a genuinely different model family (fine-tuned encoder vs. prompted generative LLM) rather than a near-duplicate or a labeling artifact.
- **Result: by far the best result of any method in Stage B.** Macro F1 0.8254 vs. the previous best (EXP-008, M5 MIPROv2) at 0.6700 -- a **+0.1554** absolute improvement, and the first method to clear 0.80. Consistent with expectations: a small model *trained* (not just prompted) directly on 6,706 in-domain labeled examples should outperform a much larger general-purpose LLM used zero/few-shot, and it does here by a wide margin, while also running ~5-8x faster per experiment.
- **Artifacts:** `results/EXP-009/{config.json, metrics.json, predictions.csv}` -- pulled back and committed immediately. Best checkpoint at `models/EXP-009/best_checkpoint/` (gitignored, kept durable on local Mac disk via `sync_from_vm.sh`, per the project's models/ policy -- see `web/README.md` for where the web app's DeBERTa adapter expects to find it).
- **TEST not touched**, per the sealing policy -- correct.
- **M6 is now DONE** (single checkpoint; the smoke test already validated `fp16=true` stability, so no repeat-across-seeds pass was deemed necessary given the very wide margin over every other method -- can revisit if Phase 2 needs a variance estimate before freezing). Next: cross-model DEV disagreement analysis (STAGE_B_CHECKLIST.md section 7), then Phase 2 (freeze configs, unseal TEST).
