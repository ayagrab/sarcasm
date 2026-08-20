# Experiment Log

Detailed technical companion to `PROJECT_SUMMARY.md`, organized by topic
and method rather than by when the work happened: environment and
infrastructure, then one section per approach (M1–M6), then the
cross-model analysis and the final configuration freeze. For the clean,
high-level narrative (results, methodology, conclusions) see
`PROJECT_SUMMARY.md`. For the project's meeting-by-meeting history, see
`docs/project_history.md`.

---

## Overview

This is a *new* phase inside the existing `sarcasm` repository — the
repository's existing (and already-implemented) work is a **sarcasm
interpretation/neutralization benchmark**: given a sarcastic tweet, an
LLM rewrites it as a sincere, non-sarcastic sentence, and the rewrite is
scored by an LLM judge / NLI model / human annotators (Alt-Test, Fleiss'
Kappa, Kruskal-Wallis, BLEU/ROUGE/PINC — see the repo's root `README.md`).
That pipeline is fully implemented (`src/generation/`,
`src/evaluation/evaluate_with_llm.py`, `src/evaluation/evaluate_with_nli.py`,
`src/postprocessing/*`) and is **not sarcasm classification** — it never
predicts sarcastic/not-sarcastic, it rewrites already-known-sarcastic
text.

The classification/detection task documented in this file was proposed at
the project's fourth supervisor meeting (2026-07-16) and scoped, at a
high level, in `docs/finetuning_plan.md`: "fine-tune a dedicated
BERT-based binary sarcasm classifier … Sarcasm Corpus V2 … GEN/HYP/RQ."
That plan covered one BERT fine-tune; the work in this file substantially
broadens it into a 6-approach comparison (classical ML, LLM zero-shot,
few-shot, structured reasoning, DSPy-optimized prompting, and a
fine-tuned Transformer encoder), reusing the same target dataset the plan
already staged.

The new classification work lives in its own clearly-separated subpackage
(`src/classification/`), its own prompts folder
(`prompts/classification/`), and its own config files (`configs/`), so it
never collides with the interpretation pipeline's code, prompts, or
config. Shared, generic infra (the `.env`/`config/settings.py` pattern,
`data/raw/` being read-only, the `python -m src.<pkg>.<script>`
convention, the pytest/mock testing style) is reused as-is.

**Status: complete.** All six approaches (M1–M6) were developed on DEV,
frozen to a single final configuration each, and evaluated exactly once
on the sealed TEST split. See `PROJECT_SUMMARY.md` for the final results
table and conclusions.

---

## Dataset

### Source and location

`data/raw/sarcasm_corpus_v2/` — **Sarcasm Corpus V2** (UC Santa Cruz,
Oraby et al.). Three CSV files, one per sarcasm-category subset:

| File | Category (documented meaning) | Rows | `sarc` | `notsarc` |
|---|---|---:|---:|---:|
| `GEN-sarc-notsarc.csv` | **General Sarcasm** — general-purpose sarcastic vs. sincere forum posts | 6,520 | 3,260 | 3,260 |
| `HYP-sarc-notsarc.csv` | **Hyperbole** — sarcasm expressed via exaggeration | 1,164 | 582 | 582 |
| `RQ-sarc-notsarc.csv` | **Rhetorical Questions** — sarcasm expressed as a rhetorical question | 1,702 | 851 | 851 |
| **Total** | | **9,386** | **4,693** | **4,693** |

Columns in every file: `class` (`sarc`/`notsarc`), `id` (integer, resets
to 1 within each file — not globally unique, so a canonical ID must
combine category + id), `text` (the post). Each category file's `text`
fields containing embedded commas/newlines are properly double-quoted
(verified with `pandas.read_csv`, not naive line counting). The three
files are each perfectly class-balanced (50/50), so the combined dataset
is too — **no class-imbalance handling is needed.**

### Data quality checks

Formalized into two reusable, re-runnable scripts:
`src/classification/data/build_canonical_dataset.py` (combines the 3 raw
files, attaches `example_id`/`category`/`source_file`/`dup_group_id`/
`label_conflict`) and `src/classification/data/audit_dataset.py`
(computes and reports the checks below, writes
`data/processed/sarcasm_v2_audit_report.json`). The audit script's
duplicate-text key normalizes whitespace runs as well as case
(`" ".join(text.strip().lower().split())`).

| Check | Result |
|---|---|
| Missing/null `text` | 0 |
| Missing/null `class` | 0 |
| `id` uniqueness within file | unique within each file (1..N) |
| `id` uniqueness across files | **not unique** — canonical `example_id` = `f"{category}-{id}"` |
| Exact duplicate `text` within a single file | 0 in every file |
| Duplicate `text` groups (normalized: case + whitespace insensitive), within **and** across files | **596 rows in 297 groups** |
| Duplicate groups with **conflicting labels** | **22 rows across several groups** (e.g. `GEN-103`, `RQ-679`..`RQ-684`) |
| Word-length distribution (whitespace tokens) | min 10, 25th pct 22, median 38, mean 48.7, 75th pct 67, max 150 |
| Degenerate very-short examples (≤2 words) | 0 |
| Degenerate very-long examples (>200 words) | 0 |
| Metadata fields beyond `class`/`id`/`text` (author, conversation, thread, timestamp, source post) | **none present in the raw files** |

Several label-conflict rows are near-consecutive RQ ids (`RQ-679` through
`RQ-684`) — likely a cluster from the same source thread annotated
inconsistently. **The 22 label-conflict rows** are a genuine annotation
inconsistency in the source corpus, not a bug in this repo. Per the
"never silently remove examples" rule, they are **kept**, tagged
`label_conflict=True` and given a shared `dup_group_id` in the canonical
dataset so they can be inspected during error analysis.

### Canonical dataset and split — decisions made

1. **No rows are dropped or deduplicated in the canonical dataset.** All
   9,386 rows are preserved, each tagged with `category`, `source_file`,
   `dup_group_id` (shared by rows with matching normalized text —
   singleton group for everything else), and `label_conflict` (bool).
   Downstream consumers (splitting, training) decide what to do with
   duplicates; the canonical dataset itself is non-destructive.
2. **Global example IDs.** Since raw `id` is only unique per file,
   canonical IDs are `f"{category}-{id}"` (e.g. `GEN-4213`) — unique,
   stable, and human-traceable back to the source file/row.
3. **Grouped splitting is required.** Because 336+ rows share text across
   category files, a purely random or purely-stratified-by-label split
   risks putting the *same underlying post* in both train and test (e.g.
   the GEN copy in train, the RQ copy in test) — the model would then be
   evaluated on text it effectively saw during training. **Decision:
   split by `dup_group_id` (normalized-text group), not by row**, so
   every row sharing a normalized text always ends up in the same split.
   Label stratification is applied at the *group* level using the
   group's label (for `label_conflict` groups, the first row's label — a
   coin flip either way, affecting only 4 groups/~0.04% of data) so the
   50/50 class balance is preserved as closely as group sizes allow. Full
   implementation: `src/classification/data/make_splits.py`.
4. **No author/conversation-level grouping** — the raw files carry none
   of that metadata, so text-based grouping is the only leakage control
   available. A known limitation of the source corpus.
5. **Split ratio:** target 70/15/15 (train/dev/test), seed `42`,
   implemented with two chained `StratifiedGroupKFold` passes (test fold
   carved out first, then train/dev from the remainder), grouped on
   `dup_group_id`. **Actual achieved split** (group sizes make an exact
   70/15/15 unreachable): **train 6,706 (71.4%) / dev 1,340 (14.3%) /
   test 1,340 (14.3%)**. Label balance held closely per split (train
   3,369/3,337 sarc/notsarc, dev 668/672, test 656/684) and category mix
   is proportionate across splits. A programmatic assertion
   (`_assert_no_group_leakage`) runs on every split build and raises if
   any `dup_group_id` ever spans more than one split — it passes.
   Persisted as canonical `data/splits/split_assignments.csv`
   (`example_id -> split`) plus materialized `train.csv`/`dev.csv`/
   `test.csv` under `data/splits/`, all reproducibly regenerated by
   `make_splits.py` from the canonical dataset + a fixed seed.

### TEST-sealing policy

The initial plan had each method (M2, M3, …) evaluate TEST immediately
after its own DEV run, reasoning that a single fixed zero-shot config has
"nothing to tune." **Corrected before any TEST evaluation actually
happened:** TEST stays completely sealed until **every** method (M2–M6)
has finished its DEV-only development and had a configuration frozen.
Rationale: even without per-method hyperparameter tuning, *cross-method*
comparisons and any judgment calls made while methods are still being
developed (e.g. deciding a prompt looks "good enough," or debugging a
parsing issue after seeing a low score) are a form of indirect tuning if
TEST is visible during that process. Sealing TEST entirely until every
method is frozen removes that risk. Concretely: every method develops on
DEV only (Phase 1) until all six are frozen, then each frozen config is
evaluated on TEST exactly once (Phase 2) — see `PROJECT_SUMMARY.md` §3.1.

### Random seeds

Global default seed for this project: **`42`** (dataset split, few-shot
example sampling, classical-baseline model, fine-tuning).

---

## Environment and Infrastructure

### Local development environment

| Item | Finding |
|---|---|
| Python | 3.10.6 (later rebuilt as a native arm64 venv — see `docs/project_structure.md`) |
| OS / machine | macOS (Darwin), Apple Silicon |
| GPU | No CUDA GPU (Apple MPS backend available) — a laptop, not a dedicated GPU compute machine |
| `dspy`, `accelerate`, `sentencepiece` | not installed by default — see `requirements-classification.txt` |

Since the local machine has no CUDA GPU, the LLM- and DSPy-based
approaches (M2–M5) and the Transformer fine-tune (M6) all needed a real
GPU machine — see "Stage B compute environment" below. The classical
baseline (M1) needs no GPU and was developed and frozen locally.

### Stage B compute environment

**Target machine:** Azure `Standard_NV24s_v3` — 24 vCPUs, 224 GiB RAM,
2x NVIDIA Tesla M60 GPUs (NVv3 family, Maxwell architecture, compute
capability 5.2). Maxwell does not support bfloat16 (needs Ampere+),
FlashAttention 2 (needs Ampere+), or modern vLLM (needs compute
capability >= 7.0) — the LLM runtime plan uses plain float16
`transformers` generation with `attn_implementation="eager"` throughout.

Supporting infrastructure, all implemented before real execution and
unit-tested with mocks/guards first (the VM-specific scripts named below
— `verify_gpu.py`, `verify_kernel.sh`, `sync_to_vm.sh`, `sync_from_vm.sh`,
`sync_cache_from_vm.sh`, and the DEV-phase chain scripts — were later
removed from the repository once the VM was no longer needed; described
here as the historical record of what was built and used):
- `scripts/verify_gpu.py` — the mandatory Stage B gate: runs
  `nvidia-smi`, records exact GPU count/model/VRAM/driver version, checks
  `torch.cuda` compute capability against BF16/FlashAttention2/vLLM
  requirements (labeled `INFORMATIONAL`, never fatal), and fails (exit 1)
  only if no CUDA GPU is visible (`REQUIRED`). Also reports an explicit
  `fp16_transformers_pipeline_ok: true/false` field.
- `src/classification/llm/local_client.py` — `LocalHFClient`, a local
  Hugging Face Transformers inference client shaped like the
  OpenAI/OpenRouter client so the existing zero-shot/few-shot/reasoning
  pipeline works unchanged against a local model (`provider="local_hf"`).
  Defaults: `dtype=torch.float16`, `attn_implementation="eager"`,
  `device_map="auto"`, `max_memory={0: "7GiB", 1: "7GiB", "cpu": "180GiB"}`,
  `low_cpu_mem_usage=True`. Hard-rejects `bfloat16`/`flash_attention_2` at
  construction time and hard-rejects loading with no CUDA GPU visible.
- `src/classification/llm/client.py` / `run_llm_classification.py` —
  extended with a `provider` parameter (`"openrouter"` default or
  `"local_hf"`); forces `concurrency=1` under `local_hf` (one shared GPU
  model instance is not safe to fan out across threads).
- `scripts/sync_to_vm.sh` / `scripts/sync_from_vm.sh` — push code/config
  to the VM and pull `results/`/`logs/`/`models/` back, respectively.

**Independently verified environment** (not just taken on trust):

| Field | Value |
|---|---|
| VM size | Standard_NV24s_v3 |
| Hostname | `dpmlgpuNC6sv32025s-0003` |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-1029-azure |
| vCPUs / RAM | 24 / ~220 GiB usable |
| Storage | `/mnt` ~1.5 TB (~1.4 TB free), `/datashare` (Azure-mounted shared FS) |
| GPUs | 2x NVIDIA Tesla M60, ~7.93 GiB VRAM each |
| Compute capability | 5.2 (Maxwell) |
| Driver | 535.230.02 |
| `nvidia-smi`-reported CUDA compatibility | 12.2 |
| Working PyTorch CUDA runtime | 11.8 (`torch==2.5.1+cu118`) |
| Installed | torch 2.5.1+cu118, transformers 5.15.0, datasets 5.0.1, sklearn 1.7.2, pandas 2.3.3, dspy 3.3.0, accelerate 1.14.0, sentencepiece |

Full snapshot saved to `environment_stage_b.txt` (`hostname`,
`python --version`, `nvidia-smi`, `pip freeze`, no secrets).
`scripts/verify_gpu.py` run for real on the VM confirmed
`fp16_transformers_pipeline_ok: True` and all three INFORMATIONAL lines
(BF16/FlashAttention2/vLLM) correctly reporting "does NOT support"
without failing the script.

**Repository-level Qwen smoke test** (production code path, real GPU,
real model): `python -m src.classification.llm.run_llm_classification
--experiment-id SMOKE-qwen-zero-shot --mode zero_shot --eval-split dev
--model Qwen/Qwen3-4B-Instruct-2507 --provider local_hf --limit 20
--concurrency 1` — model loaded in ~2s (cached), 20 DEV examples
classified in ~36s, zero label-parsing failures, confirming dataset
loading, prompt construction, the `local_hf` client, label parsing,
prediction persistence, and the evaluator all work through the real
production code path. (The smoke test's own accuracy, 0.25/macro F1
0.20, is a sampling artifact of `--limit 20` taking the first 20
unshuffled DEV rows, which happen to skew `not_sarcastic` — not a real
measurement; the full-DEV runs below use all 1,340 rows.) This exact
smoke test was re-run and reproduced identically after every environment
rebuild described below — confirmed **ten times** in total.

### Recurring infrastructure issue: ephemeral disk wipes

The VM's `/mnt` (Azure's ephemeral resource/temp disk, `/dev/sdb1`) is
**not persistent storage** — `/mnt/DATALOSS_WARNING_README.txt` (rewritten
by Azure on every boot) states explicitly that it is subject to loss with
no way to recover it. This was not obvious at the start of Stage B, since
the VM-side repo checkout, Python venv, Hugging Face model cache, and
per-example LLM disk cache had all been placed under `/mnt/vmadmin/` as
if it were durable. Over the course of the project, `/mnt` was wiped by a
VM restart **nine times** (a mix of deliberate restarts, e.g. to load a
kernel fix, and unannounced ones of unclear cause) — recovered fully
every time, with the same procedure working identically on every
occasion:

1. Verify survival: `hostname`, `uname -r` (expect the known-good kernel,
   `6.8.0-1029-azure`), `nvidia-smi` (both Tesla M60s healthy).
2. Recreate `/mnt/vmadmin/{projects,sarcasm-env,huggingface}` (occasionally
   came back root-owned, requiring `sudo mkdir`/`sudo chown` first).
3. Restore the repo via `scripts/sync_to_vm.sh`.
4. Recreate the Python venv and reinstall the **exact pinned versions**
   from `environment_stage_b.txt`'s captured `pip freeze` — never
   upgrading anything as part of a recovery.
5. Re-download `Qwen/Qwen3-4B-Instruct-2507` and
   `microsoft/deberta-v3-base` into a fresh `HF_HOME` (~9.3 GB, both
   public/ungated, done in under a minute on the VM's link).
6. Redo the DeBERTa safetensors conversion (see M6's section) — this
   doesn't survive a wipe since it lives on `/mnt`.
7. Run `scripts/verify_kernel.sh`, `scripts/verify_gpu.py`, the
   classification test suite (61/61), and the Qwen zero-shot smoke test —
   every single recovery reproduced identical results, confirming the
   environment rebuild is truly equivalent each time, not just
   "loads without crashing."
8. Restore the per-example LLM disk cache from a periodic local-Mac
   backup (`scripts/sync_cache_from_vm.sh`, pulled continuously during
   any run) so already-computed examples resume instantly instead of
   being recomputed.
9. Relaunch whatever chain script was interrupted — the chain scripts are
   resume-aware where possible (skip any step whose
   `results/<experiment_id>/metrics.json` already exists).

**Durability fixes put in place because of this recurring issue:**
- The entire Stage A/B work (log, summary, all of `src/classification/`,
  `configs/`, `data/splits/`, canonical dataset, results, tests) was
  committed to git — previously it existed only on the local Mac disk,
  uncommitted, itself a single point of failure independent of the VM.
- `scripts/sync_from_vm.sh` was added as the missing counterpart to
  `sync_to_vm.sh` — the original workflow only ever pushed code to the
  VM and never pulled results back, which is what caused one early
  experiment's raw predictions file to be lost (its aggregate metrics
  were already recorded in this log and were not lost). Going forward,
  results/logs are pulled back immediately after every experiment
  finishes, not just before a planned shutdown.
- `scripts/verify_kernel.sh` was added as a startup guard: checks
  `uname -r` against the known-good kernel and that `nvidia-smi`
  succeeds, failing loudly instead of letting an unverified kernel
  surface as a confusing failure hours into an experiment. One incident
  traced to a genuinely more serious failure mode — an auto-applied newer
  kernel package silently breaking the NVIDIA driver — was root-caused to
  GRUB's *default* boot entry never having been pinned after an earlier
  one-shot `grub-reboot` fix (which only affects the single next boot).
  Fixed permanently with `grub-set-default` + `update-grub`, verified via
  `grubenv`'s `saved_entry` — this made the known-good kernel the
  permanent default rather than needing the fix reapplied after every
  reboot.
- Model downloads (`HF_HOME`) and the per-example LLM cache remain
  intentionally on ephemeral `/mnt` — both are cheaply regenerable and
  don't need durable storage; only irreplaceable outputs (code, configs,
  splits, results, logs) need to live in git.

The most costly single incident was an unannounced outage roughly 35
minutes into DSPy MIPROv2's final sealed-TEST run (M5, no
partial-checkpoint capability for that optimizer), which cost a full
restart of that ~2h+ run from scratch — a concrete, not just
hypothetical, illustration of that method's known limitation. Everything
else survived because results/configs are committed to git immediately
after each experiment finishes, never held only on the VM's ephemeral
disk.

### One-off environment/dependency issues found and fixed

- **`litellm==1.96.1` (the pinned version) was removed from PyPI
  entirely** partway through the project — genuinely absent from the
  index, not just yanked. Substituted `litellm==1.96.2` (closest
  available); nothing downstream (dspy/MIPROv2) was affected.
- **`optuna` was missing from the pinned environment freeze.** MIPROv2
  requires it as an optional dspy dependency; it had never been captured
  in any prior `pip freeze` snapshot. Installed `optuna==4.9.0` and added
  it to `environment_stage_b.txt` for durability — a real, pre-existing
  environment gap, not caused by any infrastructure incident.
- **`microsoft/deberta-v3-base` failing to load** under the pinned
  `torch==2.5.1` (`transformers`' `check_torch_load_is_safe` guard
  requiring `torch>=2.6` because that HF repo has no safetensors weights
  upstream) — see M6's section for the fix (a local safetensors
  conversion, redone after every `/mnt` wipe since it doesn't persist).

---

## M1 — TF-IDF + Logistic Regression (classical baseline)

- **Code:** `src/classification/classical/tfidf_baseline.py`
- **Data:** canonical split (`data/splits/`), seed 42, train=6,706 /
  dev=1,340 / test=1,340
- **Configuration selection (DEV only, TEST never touched):** a 6-way
  sweep, `classifier ∈ {logreg, linear_svm}` × `tfidf_variant ∈
  {word_1_2 (word 1-2gram), char_3_5 (char 3-5gram), word_char_combo
  (FeatureUnion of both)}`, `TfidfVectorizer(min_df=2, max_features=50000)`
  for both variants, `LogisticRegression(max_iter=2000)` /
  `LinearSVC()`, both with `random_state=42`.

  | Sweep run | classifier | tfidf_variant | dev accuracy | dev macro F1 | dev sarcastic F1 |
  |---|---|---|---:|---:|---:|
  | 1 | logreg | word_1_2 | 0.7463 | 0.7462 | 0.7436 |
  | 2 | logreg | char_3_5 | 0.7470 | 0.7470 | 0.7453 |
  | 3 | logreg | word_char_combo | **0.7530** | **0.7529** | 0.7491 |
  | 4 | linear_svm | word_1_2 | 0.7410 | 0.7409 | 0.7353 |
  | 5 | linear_svm | char_3_5 | 0.7276 | 0.7276 | 0.7249 |
  | 6 | linear_svm | word_char_combo | 0.7261 | 0.7260 | 0.7192 |

  Winner by dev macro F1: **`logreg` + `word_char_combo`** — frozen and
  evaluated once, for the first and only time, on TEST.

- **Frozen final evaluation (TEST, EXP-001, run once):**
  `python -m src.classification.classical.tfidf_baseline --experiment-id EXP-001 --classifier logreg --tfidf-variant word_char_combo --eval-split test`

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.7403 |
  | Macro F1 | 0.7403 |
  | Weighted F1 | 0.7403 |
  | Sarcastic Precision / Recall / F1 | 0.7245 / 0.7576 / 0.7407 |
  | Not-sarcastic Precision / Recall / F1 | 0.7569 / 0.7237 / 0.7399 |
  | Confusion matrix (rows=gold, cols=pred, order [not_sarcastic, sarcastic]) | `[[495, 189], [159, 497]]` |

- **Runtime:** a few seconds total (fit + predict on ~9.4k short texts) —
  negligible, no GPU needed. **Cost:** $0.
- **Artifacts:** `results/EXP-001/{config.json, metrics.json, predictions.csv}`
- **Observations:** dev and test macro F1 are close (0.753 vs. 0.740,
  ~1.3 points), suggesting no meaningful overfitting to the dev-based
  configuration choice. The word+char TF-IDF combination beat either
  n-gram type alone for `logreg`, but consistently hurt `linear_svm`.
  Errors are fairly symmetric between false positives (189) and false
  negatives (159), no strong bias toward either class.
- **Conclusion:** credible, reproducible classical baseline established —
  **Macro F1 ≈ 0.740 on the frozen test set.** This is the number every
  other approach needs to beat to be worth its added cost/complexity.
- **Frozen for Phase 2 (`configs/tfidf.json`, EXP-001)** with no
  changes — this was Stage A's baseline, untouched by the rest of Stage B.

---

## M2 — Qwen3-4B zero-shot

- **Code:** `src/classification/llm/` (`client.py`, `schema.py`,
  `run_llm_classification.py`), prompt
  `prompts/classification/zero_shot_v1.txt`, config
  `configs/llm_zero_shot_qwen_local.json`.
- **Command:** `python -m src.classification.run_experiment --config configs/llm_zero_shot_qwen_local.json`
- **Config:** `provider=local_hf`, `model=Qwen/Qwen3-4B-Instruct-2507`,
  `mode=zero_shot`, `temperature=0.0` (deterministic, `do_sample=False`),
  `eval_split=dev`, seed 42.
- **Runtime:** ~37 minutes wall-clock for 1,340 examples
  (~1.6-1.9s/example, sequential — `local_hf` forces `concurrency=1`).
- **Full DEV metrics (EXP-002):**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.6440 |
  | Macro F1 | 0.6008 |
  | Weighted F1 | 0.6004 |
  | not_sarcastic: P / R / F1 (support 672) | 0.9295 / 0.3140 / 0.4694 |
  | sarcastic: P / R / F1 (support 668) | 0.5858 / 0.9760 / 0.7322 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[211, 461], [16, 652]]` |

- **Quality checks:** `n_examples=1340` matches DEV split size exactly;
  1,340 unique `example_id`s, no duplicates/missing rows; gold label
  distribution matches the canonical DEV split exactly; every prediction
  is a valid `{sarcastic, not_sarcastic}` value.
- **Investigated: predicted-label distribution is heavily skewed** — 83%
  predicted `sarcastic` vs. the true 49.9%. Checked whether this was a
  bug: predicted-sarcastic rate is uniform across categories (GEN 81.7%,
  HYP 89.2%, RQ 83.9%), ruling out a category-specific or ordering
  artifact. Manually read 5 false positives (gold `not_sarcastic`,
  predicted `sarcastic`): all are combative/rhetorical internet-debate
  text (creationism-vs-evolution arguments, political forum sparring,
  mocking rhetorical questions) that plausibly *reads* as
  sarcastic/mocking in tone even though annotated as sincere argument.
  **Conclusion: genuine zero-shot model behavior, not a pipeline bug** —
  Qwen3-4B-Instruct, given only the task definition and no examples,
  appears biased toward calling adversarial/rhetorical debate text
  "sarcastic," yielding high recall on true sarcasm (0.976) at a heavy
  cost to specificity (0.314 recall on genuinely sincere text).
- **Error analysis by category** (`results/EXP-002/analysis/dev_error_summary.json`):
  accuracy by category — GEN 0.666, RQ 0.609, HYP 0.572 (hyperbole
  hardest). Incorrect predictions average 53.9 words vs. 46.7 for correct
  ones — a mild signal that longer/more complex debate text is harder.
  Only 2 of the 22 known label-conflict rows landed in DEV — too few to
  draw a conclusion from.
- **Frozen for Phase 2** (`configs/llm_zero_shot_qwen_local.json`,
  EXP-002) — the only candidate config for this method (zero-shot has no
  hyperparameter to sweep).
- **Sealed TEST evaluation (EXP-002-TEST):** Macro F1 **0.6005** (vs. DEV
  0.6008 — almost no gap, expected since zero-shot has no config
  selection step to overfit DEV with in the first place). Quality checks:
  n=1340, no duplicate/missing `example_id`s, ID set matches
  `data/splits/test.csv` exactly, gold distribution (684/656) matches.

---

## M3 — Qwen3-4B few-shot

Two demo-selection variants compared on DEV: random and curated.

### Random demos (EXP-003)

- **Command:** `python -m src.classification.run_experiment --config configs/llm_few_shot_random_8_qwen_local.json`
- **Config:** `mode=few_shot`, `few_shot_variant=random`, `n_shots=8`,
  prompt `classification/few_shot_v1.txt`, `eval_split=dev`, seed 42.
- **Demo example IDs used** (8, randomly selected, seed 42): `RQ-502,
  GEN-3940, GEN-437, GEN-2796, GEN-1020, RQ-1304, GEN-1888, HYP-624` —
  label balance confirmed 4 `not_sarcastic` / 4 `sarcastic`.
- **Runtime:** ~1h34m wall-clock (~4.2s/example — the longer 8-shot
  prompt dominates per-call latency on the M60's no-tensor-core
  hardware).
- **Full DEV metrics:**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.6328 |
  | Macro F1 | 0.5880 |
  | Weighted F1 | 0.5876 |
  | not_sarcastic: P / R / F1 (support 672) | 0.8982 / 0.3021 / 0.4521 |
  | sarcastic: P / R / F1 (support 668) | 0.5790 / 0.9656 / 0.7239 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[203, 469], [23, 645]]` |

- **Investigated: Macro F1 (0.588) is *lower* than M2's zero-shot
  (0.601)** — surprising, since few-shot demonstrations are generally
  expected to help or at least not hurt. Checked three explanations: (1)
  demo-selection bias — no, the 8 random demos are label-balanced (4/4);
  (2) category-specific artifact — no, predicted-sarcastic rate is
  uniform across categories (GEN 82.0%, HYP 91.0%, RQ 82.3%); (3) did
  few-shot actually change model behavior, or is this noise from a
  near-identical model — compared directly against M2 on the same DEV
  set: **94.25% agreement**, i.e. the few-shot prompt barely moved the
  model's decisions, and where it did, it slightly *increased* the
  sarcastic-prediction skew. **Conclusion: not a pipeline bug** — random
  few-shot demonstrations did not meaningfully help Qwen3-4B calibrate
  away from its zero-shot bias; if anything they reinforced it slightly.

### Curated demos (EXP-004)

- **Command:** `python -m src.classification.run_experiment --config configs/llm_few_shot_curated_8_qwen_local.json`
- **Config:** same as above with `few_shot_variant=curated`.
- **Demo example IDs used** (8, curated — stratified across (category,
  label)): `GEN-5007, RQ-838, GEN-787, GEN-3116, HYP-96, RQ-159, HYP-652, GEN-5870`.
- **Runtime:** ~1h22m wall-clock (~3.7s/example).
- **Full DEV metrics:**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.5821 |
  | Macro F1 | 0.5011 |
  | Weighted F1 | 0.5005 |
  | not_sarcastic: P / R / F1 (support 672) | 0.9375 / 0.1786 / 0.3000 |
  | sarcastic: P / R / F1 (support 668) | 0.5446 / 0.9880 / 0.7021 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[120, 552], [8, 660]]` |

- **Investigated: this is the most skewed result of the four
  manual-prompt variants** — 90.4% predicted `sarcastic`, vs. 83% (M2)
  and 83.1% (M3-random). Checked: category-specific artifact — no,
  predicted-sarcastic rate is uniform across categories (GEN 90.0%, HYP
  93.4%, RQ 90.3%); curated-demo selection bias — the 8 curated demos are
  stratified across (category, label) cells by construction, not
  label-skewed by design; did the curated demos meaningfully change
  behavior, or is this the same model with noise — 91.3% agreement with
  M2, 92.2% agreement with M3-random, consistent with "same underlying
  bias, nudged slightly further." **Conclusion: not a pipeline bug** —
  curated, category/label-balanced demonstrations pushed Qwen3-4B
  *further* toward over-predicting `sarcastic` than either zero-shot or
  random few-shot, a genuine if counterintuitive result.

### M2–M4 comparison and variant selection

| Experiment | Method | Macro F1 | Accuracy | sarcastic Recall |
|---|---|---:|---:|---:|
| EXP-002 | Zero-shot | **0.6008** | **0.6440** | 0.976 |
| EXP-005 | Structured reasoning | 0.5796 | 0.6276 | 0.969 |
| EXP-003 | Few-shot (random) | 0.5880 | 0.6328 | 0.966 |
| EXP-004 | Few-shot (curated) | 0.5011 | 0.5821 | 0.988 |

**Zero-shot is the best of all four manual-prompt variants on DEV**, by a
clear and consistent margin. None of the three "smarter prompting"
variants improved on the simplest possible prompt — each one either left
the model's sarcastic-overprediction bias unchanged (reasoning) or made
it worse (both few-shot variants, curated worse than random). Between the
two few-shot variants specifically, **random (EXP-003) wins** on DEV
(0.588 vs. 0.501 Macro F1) — curated demo selection actually hurt this
task, a real if counterintuitive finding.

- **Frozen for Phase 2** (`configs/llm_few_shot_random_8_qwen_local.json`,
  EXP-003, the random variant).
- **Sealed TEST evaluation (EXP-003-TEST):** Macro F1 **0.5947** (vs. DEV
  0.5880 — close, TEST actually very slightly higher, within noise).
  Quality checks: n=1340, no duplicate/missing IDs, gold distribution
  matches.

---

## M4 — Qwen3-4B structured reasoning

- **Code:** prompt `prompts/classification/reasoning_v1.txt`, config
  `configs/llm_reasoning_qwen_local.json`.
- **Command:** `python -m src.classification.run_experiment --config configs/llm_reasoning_qwen_local.json`
- **Config:** `mode=reasoning`, `temperature=0.0`, `eval_split=dev`, seed 42.
- **Runtime:** ~1h02m wall-clock (~2.7-2.9s/example — faster than either
  few-shot variant, slower than plain zero-shot, consistent with a short
  prompt but a longer generated completion, i.e. the reasoning trace).
- **Full DEV metrics (EXP-005):**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.6276 |
  | Macro F1 | 0.5796 |
  | Weighted F1 | 0.5791 |
  | not_sarcastic: P / R / F1 (support 672) | 0.9023 / 0.2887 / 0.4374 |
  | sarcastic: P / R / F1 (support 668) | 0.5751 / 0.9686 / 0.7217 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[194, 478], [21, 647]]` |

- **Investigated:** same skew pattern as every other Qwen variant — 83.4%
  predicted sarcastic overall, uniform across categories (GEN 82.6%, HYP
  90.4%, RQ 84.7%), ruling out a category-specific artifact. 94.6%
  agreement with M2 (zero-shot) — the highest agreement of any few-shot/
  reasoning variant against zero-shot, meaning explicit step-by-step
  reasoning barely moved the model's decisions at all, and where it did,
  it landed almost exactly on the same skew. **Conclusion: not a pipeline
  bug** — prompting Qwen3-4B to reason step-by-step before answering did
  not meaningfully change its behavior relative to direct zero-shot
  prompting; the model's underlying tendency to read adversarial/
  rhetorical text as sarcastic appears robust to prompting strategy (see
  the M2–M4 comparison table in M3's section above).
- **Frozen for Phase 2** (`configs/llm_reasoning_qwen_local.json`,
  EXP-005) — the only candidate config for this method.
- **Sealed TEST evaluation (EXP-005-TEST):** Macro F1 **0.5758** (vs. DEV
  0.5796 — close, consistent with M2/M3's pattern). Quality checks: n=1340,
  no duplicate/missing IDs, gold distribution (684/656) matches.

---

## M5 — DSPy-optimized prompting

Three variants compared on DEV: `Predict` (unoptimized baseline),
`BootstrapFewShot`, and `MIPROv2`.

### `Predict` — unoptimized baseline (EXP-006)

- **Code:** `src/classification/dspy_pipeline/` (`signatures.py`,
  `run_dspy.py`), config `configs/dspy_predict.json`.
- **Command:** `python -m src.classification.run_experiment --config configs/dspy_predict.json`
- **Config:** `provider=local_hf`, `optimizer=predict` (unoptimized
  `dspy.Predict` over the `SarcasmClassification` signature — plain
  input/output field wrapper, no bootstrapped demos, no prompt
  optimization), `temperature=0.0`, `eval_split=dev`, seed 42.
- **Runtime:** 56m49s wall-clock (~2.5s/example).
- **Full DEV metrics:**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.6799 |
  | Macro F1 | **0.6619** |
  | Weighted F1 | 0.6616 |
  | not_sarcastic: P / R / F1 (support 672) | 0.8384 / 0.4479 / 0.5839 |
  | sarcastic: P / R / F1 (support 668) | 0.6218 / 0.9132 / 0.7398 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[301, 371], [58, 610]]` |

- Predicted-sarcastic rate 73.2% overall, roughly uniform across
  categories (GEN 70.8%, HYP 80.7%, RQ 77.0%) — less skewed than any of
  the four manual-prompt variants (83-90%). 86.6% agreement with M2
  (zero-shot manual prompt) — same underlying model, same general
  direction, but DSPy's structured signature-based prompt construction
  clearly changes behavior enough to matter. **This was the best result
  of any method at this point**, ahead of every manual-prompt variant,
  using DSPy's own default template with no optimization at all.

### `BootstrapFewShot` (EXP-007)

- **Config:** `optimizer=bootstrap_few_shot`, `max_bootstrapped_demos=4`,
  `max_labeled_demos=8`, `max_rounds=1`, `trainset_sample_size=150`
  (samples 150 of TRAIN's rows to search for valid bootstrapped demos;
  confirmed via `build_program()`: `BootstrapFewShot.compile()` is called
  with `trainset` only, no `valset` — no DEV/TEST leakage into the
  bootstrap search itself).
- **Runtime:** 2h48m24s wall-clock for compile + full-DEV eval combined —
  much longer than EXP-006's 56m49s. **Root cause, confirmed live via
  `py-spy` process inspection during the run:** once bootstrapped demos
  are found, every subsequent classification call embeds up to 8
  few-shot demos directly in the prompt (~1,200-1,400 tokens vs. EXP-006's
  short zero-shot prompt), and Tesla M60 has no flash-attention support
  (`attn_implementation='eager'`, O(n²) cost), so per-example cost during
  the DEV eval phase rose to ~7.5s/example. (DSPy's own bootstrap-phase
  progress bar writes to a pipe that block-buffers and can appear frozen
  on-screen for tens of minutes while the process is still genuinely
  computing — confirmed not a hang by installing `py-spy` on the VM
  and reading the live Python call stack, including the evaluation
  loop's row index advancing across repeated snapshots. Worth reusing
  this technique for any future silent-progress-bar situation on this
  VM.)
- **Full DEV metrics:**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.6664 |
  | Macro F1 | 0.6406 |
  | Weighted F1 | 0.6403 |
  | not_sarcastic: P / R / F1 (support 672) | 0.8641 / 0.3973 / 0.5443 |
  | sarcastic: P / R / F1 (support 668) | 0.6072 / 0.9371 / 0.7369 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[267, 405], [42, 626]]` |

- Predicted-sarcastic rate roughly uniform across categories (GEN 75.6%,
  HYP 82.5%, RQ 78.2%). 86.7% agreement with EXP-006. `compiled_program.json`
  saved with the 8 selected demos (4 bootstrapped + up to 8 labeled) —
  manually inspected a couple, both correctly labeled `not_sarcastic`
  examples of ambiguous/argumentative (but sincere) text. **Result:
  BootstrapFewShot slightly *underperforms* the unoptimized `Predict`
  baseline** (0.6406 vs. 0.6619) — echoing the M2-M4 finding that adding
  few-shot demonstrations does not help this model/corpus, now confirmed
  a second time with DSPy's automatic (not hand-curated) demo selection.

### `MIPROv2` (EXP-008)

- **Config:** `optimizer=mipro_v2`, `optimizer_config={"auto": "light",
  "trainset_sample_size": 150, "valset_sample_size": 100}`
  (`num_trials` is fixed by the `auto="light"` preset regardless of
  `valset_sample_size`, and the dominant fixed cost is the final
  full-DEV eval loop, which doesn't depend on `valset_sample_size` either).
- **A real code bug found and fixed before this run could complete:**
  Trial 1 of MIPROv2's optimization loop initially failed with
  `ValueError: Field types must be types, but received:
  ForwardRef("Literal['sarcastic', 'not_sarcastic']")…`. **Root cause:**
  `src/classification/dspy_pipeline/signatures.py` had `from __future__
  import annotations` at module level, but `SarcasmClassification` is
  defined *inside* `build_signature()`, a local function scope. Under
  postponed evaluation, its `label: Literal[...]` annotation is stored as
  a string, and pydantic/dspy resolve such strings against the *module's*
  globals — which don't include the function-local `from typing import
  Literal` — so the annotation stayed an unresolved `ForwardRef`. This
  never surfaced in `Predict` or `BootstrapFewShot` because neither calls
  `Signature.with_instructions()`; MIPROv2 is the first optimizer to
  rebuild the signature with new instructions, which re-validates field
  types. **Fix:** removed `from __future__ import annotations` from
  `signatures.py` (nothing else in that short file needed postponed
  evaluation) — confirmed `build_signature().with_instructions(...)` then
  works, full test suite (61/61) still passes.
- **Runtime (DEV run):** ~1h13m for the optimization phase (bootstrap +
  13 trials, including periodic full-100-valset checkpoint evals) +
  ~1h16m for the final full-1,340-DEV eval (~3.4s/example — notably
  faster than BootstrapFewShot's ~7.5s/example, explained below) =
  **~2h29m total**.
- **What MIPROv2 chose (DEV run):** out of 3 proposed instruction
  candidates and 6 bootstrapped few-shot demo sets, the winning
  combination was the **original/default instruction, unchanged**
  ("Classify whether an English sentence is sarcastic.") **+ a compact
  4-demo set** — MIPROv2 tried rewriting the instruction but the default
  instruction paired with a well-chosen small demo set won out over every
  rewritten-instruction candidate. This is why the final eval ran at
  ~3.4s/example rather than BootstrapFewShot's ~7.5s/example: 4 short
  demos in the prompt, not up to 8.
- **Full DEV metrics:**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.6843 |
  | Macro F1 | **0.6700** |
  | Weighted F1 | 0.6698 |
  | not_sarcastic: P / R / F1 (support 672) | 0.8201 / 0.4747 / 0.6013 |
  | sarcastic: P / R / F1 (support 668) | 0.6288 / 0.8952 / 0.7387 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[319, 353], [70, 598]]` |

- Predicted-sarcastic rate 71.0% overall, roughly uniform across
  categories (GEN 69.9%, HYP 80.1%, RQ 69.0%). 86.4% agreement with
  EXP-006, 87.5% with EXP-007. **Result: MIPROv2 is the best DSPy result**
  — Macro F1 0.6700 vs. EXP-006's 0.6619 (+0.0081) and EXP-007's 0.6406 —
  achieved with an even *smaller* few-shot set (4 demos) than
  BootstrapFewShot's up-to-8, suggesting the gain comes from smarter
  *selection* of which demos to use (via the optimization loop's
  minibatch scoring), not from more demos or a cleverer instruction.

### Frozen configuration and sealed TEST evaluation

- **Frozen for Phase 2** (`configs/dspy_mipro_v2.json`, EXP-008) — the
  DEV-best of the three DSPy variants, accepting a known TEST-time cost:
  `build_program()` always recompiles from scratch (no "load compiled
  program" path exists), so the TEST run costs a full re-optimization,
  not a quick eval.
- **Sealed TEST evaluation (EXP-008-TEST):** the optimization search (13
  trials, `auto="light"`) proposed 3 candidate instructions; the winning
  combination this time was a different instruction/demo-set pairing
  than the DEV run found (MIPROv2's search is not perfectly deterministic
  run to run even with a fixed seed) — best full-eval score 74.0. The
  subsequent final-program evaluation against the full 1,340-row sealed
  TEST split then ran for **~3h17m** — far longer than initially
  estimated by extrapolating from the optimization phase's per-call
  latency, confirmed genuinely still computing (not hung) via `nvidia-smi`
  GPU utilization and the process's live state partway through. This run
  needed to be restarted once from scratch after an infrastructure
  interruption (see "Recurring infrastructure issue," above) before
  completing.
  - **Result:** Macro F1 **0.6681** (accuracy 0.6866, weighted F1
    0.6665) — vs. DEV 0.6700, a ~0.2pt gap, consistent with the close
    DEV/TEST tracking seen for every other method. Confusion matrix
    `[[302, 382], [38, 618]]` (not_sarcastic recall 44.2% vs. sarcastic
    recall 94.2%) — the same systematic sarcastic-over-prediction bias
    documented for every LLM-based method, present here too despite M5
    being DEV-best among the LLM methods.
  - Quality checks: n=1340, 0 duplicate/missing `example_id`s, ID set
    matches `data/splits/test.csv` exactly, gold distribution (684/656)
    matches.
- **M5's exact optimized prompt** (`results/EXP-008-TEST/compiled_program.json`,
  the authoritative saved program state — `run_dspy_experiment` calls
  `program.save(...)` for any non-`predict` optimizer): quoted in full,
  instruction text and all 4 few-shot demonstrations, in
  `PROJECT_SUMMARY.md` §6.1.

---

## M6 — Fine-tuned `microsoft/deberta-v3-base`

### Blockers found and fixed before the first real run

- **DeBERTa checkpoint failing to load.** `microsoft/deberta-v3-base`'s
  HF repo has no `model.safetensors` upstream, and pinned `torch==2.5.1`
  fails `transformers`' `check_torch_load_is_safe` guard (which requires
  `torch>=2.6` to load a bare `.bin` file). **Fix:** load the cached
  `pytorch_model.bin` directly with `torch.load(..., weights_only=True)`
  (safe here — Microsoft's own official repo, not third-party), convert
  to `model.safetensors` with `safetensors.torch.save_file`, and write it
  into the same HF cache snapshot directory alongside the original files.
  Confirmed `AutoModelForSequenceClassification.from_pretrained(...,
  use_safetensors=True)` then succeeds, all 198 backbone weights loading
  cleanly (only the MLM head keys are UNEXPECTED and
  `classifier.*`/`pooler.*` MISSING/newly-initialized — both expected and
  correct for fine-tuning a base checkpoint into a 2-class classifier).
  No code changes needed beyond the conversion — the configs already
  specify `use_safetensors: true`. Since `/mnt` is ephemeral, this
  conversion has to be redone after every `/mnt` wipe.
- **`TrainingArguments.__init__() got an unexpected keyword argument
  'warmup_ratio'`.** `transformers==5.15.0` removed the standalone
  `warmup_ratio` parameter — `warmup_steps` is now overloaded to accept
  either an absolute step count (`>=1`) or a fraction of total training
  steps (`<1`). **Fix:** `finetune.py`'s `TrainingArguments(...)` call
  now passes `warmup_steps=config.warmup_ratio` instead of
  `warmup_ratio=config.warmup_ratio` — the config field name/semantics
  are unchanged, only the kwarg it's forwarded under.
- **`eval_loss: nan` every epoch, final DEV predictions collapsed to a
  single class (macro F1 ~0.33, chance level).** Root cause: the model
  load never specified a dtype, and `microsoft/deberta-v3-base`'s
  `pytorch_model.bin` on the HF Hub is itself stored in **float16**
  (confirmed via a raw `torch.load` dtype check) — so the model trained
  natively in fp16 with no fp32 master weights, causing gradient
  underflow/NaN within a handful of steps regardless of the Trainer's own
  `fp16` flag (this also explains an earlier `ValueError: Attempting to
  unscale FP16 gradients` crash — the `GradScaler` expects fp32 master
  weights to unscale into, and the model had none). **Fix:** added
  `dtype=torch.float32` to the `from_pretrained(...)` call — forces fp32
  weights regardless of the checkpoint's on-disk dtype, matching standard
  fine-tuning practice. Verified via an ad hoc `Trainer` run (logging
  every step) that losses are sane and gradients well-behaved under both
  `fp16=false` and `fp16=true` — `fp16=true` was restored in both configs
  for training speed on the M60s, now confirmed genuinely stable.

### Full training run (EXP-009)

- **Command:** `python -m src.classification.run_experiment --config configs/transformer_deberta_v3_base.json`
- **Config:** full TRAIN split (`n_train=6706`), `max_length=128`,
  `learning_rate=1e-5`, `train_batch_size=16`, `eval_batch_size=32`,
  `num_epochs=5` (early stopping `patience=2` on DEV Macro F1),
  `warmup_ratio=0.1`, `weight_decay=0.01`, `fp16=true`,
  `use_safetensors=true`, `use_fast_tokenizer=false`, seed 42.
- **Runtime:** ~22 minutes wall-clock for the full run, including
  per-epoch DEV eval — dramatically faster than any M2-M5 LLM-based
  method (each 1-3 hours), as expected for a 184M-parameter encoder vs. a
  4B-parameter LLM doing generative inference.
- **Training dynamics:** DEV Macro F1 by epoch — 1: 0.8193, 2: **0.8254
  (best)**, 3: 0.8178 (regression), then early stopping triggered after
  epoch 4 also failed to improve; best checkpoint (epoch 2) restored
  automatically. Training loss kept decreasing after epoch 2 while DEV
  metrics plateaued/regressed — classic overfitting onset on a
  6,706-example TRAIN set with a 184M-parameter encoder, exactly what
  early stopping exists to catch.
- **Full DEV metrics (best checkpoint, epoch 2):**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.8254 |
  | Macro F1 | **0.8254** |
  | Weighted F1 | 0.8254 |
  | not_sarcastic: P / R / F1 (support 672) | 0.8211 / 0.8333 / 0.8272 |
  | sarcastic: P / R / F1 (support 668) | 0.8298 / 0.8174 / 0.8235 |
  | Confusion matrix [gold rows, pred cols, order (not_sarcastic, sarcastic)] | `[[560, 112], [122, 546]]` |

- **Quality checks:** n=1340, 1,340 unique `example_id`s, no duplicates,
  ID set matches `data/splits/dev.csv` exactly, gold distribution
  matches. Predicted distribution (682 not_sarcastic / 658 sarcastic) is
  close to gold, not collapsed to one class. Predicted-sarcastic rate by
  category: GEN 50.4%, HYP 54.8%, RQ 40.3% — some spread but not wildly
  skewed. Agreement with M5's `Predict` baseline (EXP-006): 68.1%;
  agreement with M5's MIPROv2 (EXP-008): 70.8% — meaningfully different
  from both, consistent with a genuinely different model family
  (fine-tuned encoder vs. prompted generative LLM).
- **Result: by far the best result of any method.** Macro F1 0.8254 vs.
  the previous best (M5 MIPROv2) at 0.6700 — a **+0.1554** absolute
  improvement, the first method to clear 0.80. Consistent with
  expectations: a small model *trained* (not just prompted) directly on
  6,706 in-domain labeled examples outperforms a much larger
  general-purpose LLM used zero/few-shot, and it does here by a wide
  margin, while also running ~5-8x faster per experiment.
- **Artifacts:** `results/EXP-009/{config.json, metrics.json,
  predictions.csv}`. Best checkpoint at `models/EXP-009/best_checkpoint/`
  (gitignored, kept durable on local disk via `sync_from_vm.sh`).
- Single checkpoint (one seed) — the smoke test already validated
  `fp16=true` stability, and the margin over every other method was wide
  enough that a multi-seed pass wasn't judged necessary before freezing
  (see `PROJECT_SUMMARY.md`'s Future Work for a multi-seed variance
  estimate as a follow-up).

### Frozen configuration and sealed TEST evaluation

- **Frozen for Phase 2** (`configs/transformer_deberta_v3_base.json`,
  EXP-009) — the only candidate run, and the overall best result of the
  project; **`deberta` is the recommended production model** (see
  `PROJECT_SUMMARY.md` §11).
- Evaluating the frozen configuration on TEST required a dedicated
  eval-only path: `finetune.py`'s `finetune_and_evaluate` always calls
  `trainer.train()` — reusing it for the TEST step would have **silently
  trained a second model from scratch** rather than evaluating the actual
  frozen checkpoint that was reviewed and recorded. **Fix:** added
  `scripts/eval_frozen_checkpoint.py`, a standalone eval-only script that
  loads `models/EXP-009/best_checkpoint` directly (forcing
  `dtype=torch.float32`, same fix as training) and evaluates it via the
  same `save_experiment_artifacts`/`compute_metrics` utilities every
  other approach uses — verified on a 20-row smoke slice before the real
  run.
- **Sealed TEST evaluation (EXP-009-TEST):** Macro F1 **0.8209** (vs. DEV
  0.8254 — a tiny, unremarkable gap, confirming M6 generalizes
  essentially as well on genuinely unseen data as it appeared to on DEV).
  Predicted distribution (670/670) is close to perfectly balanced, closer
  to gold (684/656) than any other method's TEST predictions —
  consistent with the DEV-time finding that M6 (trained on labels)
  doesn't share the LLM methods' systematic sarcastic-over-prediction
  bias. Quality checks: n=1340, no duplicate/missing IDs, gold
  distribution matches.

---

## Cross-Model Analysis

### DEV-time analysis

All 9 DEV-evaluated experiments (M2 through M6's three DSPy variants and
final config, plus M1's frozen config re-scored on DEV as a reference
point — M1 was frozen back in Stage A with only a TEST-split prediction
file; re-running the identical frozen config with `eval_split=dev` gives
a comparable DEV prediction file without touching or invalidating the
sealed TEST result: **M1 DEV Macro F1 0.7529**, already beating every
LLM-based method on DEV, second only to M6). Full per-example table:
`results/cross_model_dev_analysis.csv`.

- **Pairwise agreement:** the LLM-based methods (M2, M3-random,
  M3-curated, M4, M5-Predict/Bootstrap/MIPROv2) cluster tightly together
  (86-95% pairwise agreement) — they mostly make the *same* mistakes as
  each other, consistent with sharing the same underlying Qwen3-4B model
  and prompt family. M1 and M6 (the two non-LLM, "trained-on-labels"
  methods) agree with each other far more (82.3%) than either agrees
  with any LLM method (58-71%) — a real methodological split, not noise.
- **Distribution of `n_models_correct` (0-9) per example:** 491/1340
  (36.6%) are correctly classified by **all 9** methods (genuinely easy);
  64/1340 (4.8%) are wrong for **every single method**, including M6's
  0.8254 — the dataset's genuinely hardest/most ambiguous rows, not
  fixable by any approach tried. A further 60 examples are correct for
  only 1/9 methods.
- **Systematic FP/FN bias — the most actionable finding:** every
  LLM-based method (M2, M3, M4, M5) is heavily FP-skewed (over-predicts
  "sarcastic"): e.g. M2 zero-shot FP=461 vs. FN=16; M3-curated FP=552 vs.
  FN=8; M5-MIPROv2 (best LLM method) FP=353 vs. FN=70. M1 and M6 are far
  more balanced (M1: FP=157/FN=174; M6: FP=112/FN=122). This is a
  **systematic bias of prompting Qwen3-4B for this task**, not fixable by
  prompt engineering alone, and the main reason M1/M6 outperform every
  LLM variant by such a wide margin.
- **Category breakdown:** every method does worst on HYP (hyperbole) and
  best on RQ (rhetorical questions) except M1, which is fairly flat
  across categories. M6 leads in every category (GEN 0.828, HYP 0.783, RQ
  0.843) — the margin over the next-best method per category is 5-9
  points, so M6's overall lead isn't driven by one easy category.
- **The 22 label-conflict rows:** only 2 of the 22 fall in the DEV split
  — too few for a statistically meaningful conclusion, but directionally
  every method scores lower on them than on the rest of DEV.
- **Confidence calibration** (M1 and M6, the only methods with
  per-example confidence): both reasonably well-calibrated — accuracy
  rises monotonically (M1: 55.9% at conf<0.6 up to 96.6% at conf>0.9; M6:
  56.9% up to 89.9%). M6's confidence distribution is far more
  concentrated at the top end (900/1340 examples >0.9 confidence) —
  expected for a fine-tuned transformer's softmax vs. a linear model's
  probability estimates, and directly usable for a future
  confidence-based human-review-routing feature.

### TEST-time analysis

Mirrors the DEV-time analysis over all 6 frozen configs'
sealed TEST predictions (`results/cross_model_test_analysis.csv`) — every
finding from DEV holds on TEST too, itself a finding (no DEV-only
artifact):

- **The four LLM-based methods (M2–M5) cluster tightly together**
  (88–95% pairwise agreement). **M1 and M6 agree with each other far more
  (80.6%) than either agrees with any LLM method (62–71%)**.
- **Systematic sarcastic-over-prediction bias, confirmed on TEST:** M2
  FP=465/FN=18, M3 FP=470/FN=19, M4 FP=489/FN=17, M5 (best-calibrated of
  the four, still skewed) FP=382/FN=38. M1 FP=189/FN=159, M6 FP=127/FN=113.
  Every LLM variant's sarcastic-class recall is 94–97%, but at the cost
  of not-sarcastic recall crashing to 28–44%.
- **Distribution of `n_models_correct` (0–6) per TEST example:** 568/1,340
  (42.4%) correctly classified by all 6 methods; 89/1,340 (6.6%) wrong for
  every single method, including M6's 0.8209 — the dataset's hardest
  rows, not fixed by any approach tried.
- **Category breakdown:** M6 leads in every category (GEN 0.833, HYP
  0.704, RQ 0.854) — an 8–15 point margin over the next-best method (M1)
  per category.
- **The 8 label-conflict rows in TEST/DEV/TRAIN** (of 22 total, split as
  2/2/4): every method scored exactly 50% on the 2 TEST rows — too few to
  be statistically meaningful, but directionally consistent with these
  rows being inherently ambiguous/contradictorily-labeled by construction.

---

## Final Configuration Freeze (Phase 2)

Per the project's Phase 2 policy: select exactly one final configuration
per method, record why, mark FROZEN. The DEV-best config is frozen for
every method (no accuracy/cost tradeoff taken) — including M5, where
MIPROv2 costs ~1.5h more on TEST than plain `Predict` for +0.008 DEV
Macro F1: the best result is kept regardless of that cost.

| Method | Frozen config | Experiment ID | Why |
|---|---|---|---|
| M1 | `configs/tfidf.json` | EXP-001 | Already frozen in Stage A, independent of Stage B |
| M2 | `configs/llm_zero_shot_qwen_local.json` | EXP-002 | Only candidate (zero-shot has no hyperparameter to sweep) |
| M3 | `configs/llm_few_shot_random_8_qwen_local.json` | EXP-003 | Random beats curated on DEV (+0.088 absolute Macro F1) |
| M4 | `configs/llm_reasoning_qwen_local.json` | EXP-005 | Only candidate |
| M5 | `configs/dspy_mipro_v2.json` | EXP-008 | Best of 3 DSPy variants on DEV, cost accepted |
| M6 | `configs/transformer_deberta_v3_base.json` | EXP-009 | Only candidate, and the overall best result — **chosen production model** |

**Final TEST scoreboard** (every method's single, one-shot, sealed
evaluation): M1 0.7403, M2 0.6005, M3 0.5947, M4 0.5758, M5 0.6681, M6
**0.8209 (best)**. See `PROJECT_SUMMARY.md` for the full results table,
comparison, error analysis, and conclusions.

---

## Demo web app (built, then excluded from the final submission)

A FastAPI + Next.js demo app (Simple Mode + a Research Mode comparing all
six methods side by side) was built and tested on top of this project's
classification code — fully working, its own test suite passing,
consuming the exact frozen inference configurations above through a
small adapter layer. It was ultimately excluded from the final submission
to keep scope focused on the research pipeline itself; the classification
code and results it consumed are unaffected.

---

## Part III — SIGN Generalization (new phase)

**Status (2026-08-20): Phase 0 (audit/planning) and Phase 1 (foundation)
complete.** Master roadmap, phase-by-phase status, and (as they land)
results: **[`SIGN_GENERALIZATION_PLAN.md`](SIGN_GENERALIZATION_PLAN.md)**.

**Phase 1 summary** (full detail in the plan doc's Phase 1 entry): built
a family-aware SIGN loader (`src/sign/data/load_sign.py` +
`family_utils.py`) on top of SIGN's official train/dev/test files, which
were already present in this repo at
`data/raw/original_{train,dev,test}_dataset.csv` (no new download
needed). Verified real counts: train 12,000 pairs / 2,292 unique-text
families (2,185 with exactly 5 interpretations); dev 1,500 pairs / 270
families (240 clean); test **1,470** pairs (30 short of the official
1,500) / 265 families (237 clean) — the test-count shortfall is a
property of the raw file itself (independently corroborated by Part I's
`data/processed/clean_sarcastic_sentences.csv`, which dedups the same
file to the same 265), not a bug in this phase's parsing. Added a
deterministic, family-leakage-safe sampling module and a family-aware
metrics module (`src/sign/family_eval/metrics.py`: original detection
rate, interpretation non-sarcasm rate, pairwise contrastive accuracy,
strict/soft family accuracy) ahead of schedule, since both are pure
computation with no model dependency. 37 new tests added
(`tests/test_sign_*.py`), 187/187 passing repo-wide, zero regressions to
the Part II suite.

This is a new, additive phase — it reuses M1–M6's frozen configs/
predictions/checkpoints above as **read-only inputs** and does not modify
anything recorded earlier in this file. Every experiment in this phase
gets a distinct `EXP-SIGN-###` ID (never `EXP-0##`, which is reserved for
the Part II record above) and writes to `results/sign/`, never
`results/EXP-0##/`. Detailed per-experiment entries (ID, research
question, exact data usage, config, seed, metrics, runtime, artifacts,
conclusion — per this file's established format) will be appended below
as each experiment in `SIGN_GENERALIZATION_PLAN.md`'s phase list actually
runs.

**Phase 2 — Dataset characterization (2026-08-20).** Compared Dataset A
(9,386) vs. SIGN originals (2,827) vs. SIGN interpretations (14,970),
always kept separate. Headline findings (full detail:
`SIGN_GENERALIZATION_PLAN.md` §6, artifacts:
`results/sign/characterization/`): Dataset A is forum-post-length
(48.7 words/example avg) vs. SIGN's tweet-length (~12–14 words); SIGN's
raw text is essentially all-lowercase and punctuation-free (~0.1–0.25%
have any uppercase char) vs. Dataset A (75.6% do) — flagged as a
methodological risk for Phase 3; VADER sentiment shows a clean,
theoretically-expected split (SIGN originals mean +0.27, sincere
interpretations mean −0.17 — the sarcastic-tweet-is-surface-positive /
interpretation-reveals-negative-truth structure showing up in an
off-the-shelf tool); PCA/UMAP of `all-MiniLM-L6-v2` embeddings shows
Dataset A visually separated from SIGN, while SIGN originals and
interpretations overlap heavily with *each other* — a first hint that
telling SIGN originals from their own interpretations may be harder than
telling SIGN from Dataset A at all.

**Phase 3 — Dataset-origin classification (2026-08-20).**
`EXP-SIGN-001`/`EXP-SIGN-002`: TF-IDF (word 1-2gram + char_wb 3-5gram,
matching M1/EXP-001's winning vectorizer) + `LogisticRegression(class_weight="balanced")`,
train on {Dataset A train (6,706) + SIGN train, all roles (14,292)}, eval
on {Dataset A test (1,340) + SIGN test, all roles (1,735)}. Raw text:
**accuracy 0.9561, macro F1 0.9555**. Case+punctuation-normalized text
(both corpora, isolating content signal from the Phase 2 formatting
confound): **accuracy 0.9242, macro F1 0.9235** — only a 3.2-point drop,
meaning the separability is mostly genuine topical/lexical/length
difference, not the formatting artifact alone. Strong, direct evidence
of domain shift between the two corpora, going into Phase 4 as the
hypothesis to test against real sarcasm-detection performance.
Artifacts: `results/sign/EXP-SIGN-001/`, `results/sign/EXP-SIGN-002/`.

**Next: Phase 4 (zero-transfer to SIGN) — blocked on the Azure VM for
the M2–M5 legs.** Not yet started.
