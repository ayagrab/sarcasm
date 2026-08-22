# Project Structure — Full Reference

A complete, folder-by-folder and file-by-file walkthrough of everything in
`sarcasm/`: what it is, what it does, and how it's used. For a quicker
overview and setup instructions, see the main `README.md`. For the full
results and methodology, see `PROJECT_SUMMARY.md`.

---

## Top-level layout

```text
sarcasm/
├── config/          # project-wide settings and model-ID constants
├── configs/         # one JSON config per Stage B (classification) experiment
├── data/            # datasets and result files (no code)
├── docs/            # research documentation
├── models/          # trained checkpoints (gitignored, not in version control)
├── prompts/         # every prompt template, as plain .txt files
├── results/         # per-experiment metrics/predictions (Stage B)
├── scripts/         # GPU-VM workflow: sync, verification, experiment chains
├── src/             # all Python code, one subfolder per pipeline stage
├── tests/           # pytest suite (no real API calls, no model/GPU needed)
├── .env             # local API keys (not committed to git)
├── .gitignore
├── conftest.py      # makes `config`/`src` importable from tests/
├── README.md
├── PROJECT_SUMMARY.md   # full project results, methodology, conclusions (all three parts)
├── Sarcasm_Project_Report.docx  # full formal write-up, submission-ready
├── requirements.txt
├── requirements-classification.txt
└── requirements-dev.txt
```

The project has two tracks:
1. **Interpretation pipeline**: evaluate how well LLMs rewrite sarcastic
   text as sincere statements, and whether an LLM judge can replace human
   annotators for scoring that. Complete.
2. **Detection ("Stage B")**: given a short English text, predict whether
   it's sarcastic at all. Six approaches compared (classical ML, LLM
   zero/few-shot/reasoning prompting, DSPy-optimized prompting, a
   fine-tuned Transformer encoder), each evaluated once on a sealed TEST
   split. Complete — see `PROJECT_SUMMARY.md`.

---

## `config/`

Project-wide configuration, imported by nearly every script instead of
hardcoding paths, keys, or model names.

- **`__init__.py`** — empty; marks `config/` as an importable Python package.
- **`models.py`** — the single source of truth for model identifiers:
  - `GENERATION_MODELS`: `{"gemini": ..., "nvidia": ..., "liquid": ...}` —
    the 3 models used to generate interpretations.
  - `JUDGE_MODELS`: `{"openrouter_llm_judge": ..., "nli": ...}` — the LLM
    judge model and the NLI model used for automatic evaluation.
- **`settings.py`** — the `Settings` dataclass (instantiated once as
  `settings`) for the **interpretation pipeline**: data folder paths,
  `prompts_dir`, API keys loaded from `.env`, and default model IDs
  pulled from `config/models.py`.
- **`classification_settings.py`** — the equivalent `Settings` dataclass
  for the **Stage B / classification pipeline**, kept deliberately
  separate so the two phases never share or accidentally overwrite each
  other's configuration: canonical dataset paths, `data/splits/` paths,
  `train_frac`/`dev_frac`/`test_frac` (0.70/0.15/0.15) and the split seed
  (42), `results_dir`, `models_dir`.

---

## `configs/`

One JSON file per Stage B experiment or method — every model/prompt/
hyperparameter/optimizer choice lives here, never hardcoded in `src/`, so
switching configuration never requires touching code. Grouped by method:

| Method | DEV config(s) | Frozen for TEST | TEST config |
|---|---|---|---|
| M1 — TF-IDF + LR | `tfidf.json` | ✓ (Stage A) | *(same config, `eval_split=test`)* |
| M2 — Qwen zero-shot | `llm_zero_shot_qwen_local.json` (+ `llm_zero_shot.json`, OpenRouter variant) | ✓ | `EXP-002-TEST.json` |
| M3 — Qwen few-shot | `llm_few_shot_random_8_qwen_local.json` (frozen), `llm_few_shot_curated_8_qwen_local.json` (DEV-only comparison) (+ OpenRouter equivalents) | ✓ (random variant) | `EXP-003-TEST.json` |
| M4 — Qwen structured reasoning | `llm_reasoning_qwen_local.json` (+ `llm_reasoning.json`) | ✓ | `EXP-005-TEST.json` |
| M5 — DSPy | `dspy_predict.json`, `dspy_bootstrap_few_shot.json`, `dspy_mipro_v2.json` (frozen) | ✓ (MIPROv2) | `EXP-008-TEST.json` |
| M6 — Fine-tuned DeBERTa | `transformer_deberta_v3_base.json` (+ `_smoke.json` for the crash/NaN smoke test) | ✓ | *(evaluated directly from the checkpoint via `scripts/eval_frozen_checkpoint.py`, no separate TEST config)* |
| *(not run)* | `transformer_roberta_base.json` | — | — |

Each TEST config's `_note` field records exactly why it exists and that
it shares every hyperparameter with its DEV counterpart (only
`eval_split`/`experiment_id` differ) — see `PROJECT_SUMMARY.md` §3.1 for
the TEST-sealing policy these configs follow.

---

## `data/`

Datasets and result files only — no code belongs here.

### `data/alt_test/`
The raw data behind the interpretation pipeline's Alt-Test result (see
`docs/alt_test_reference.md` and `src/postprocessing/run_alt_test.py`).
- **`humans_annotations.json`** — the 3 human annotators' (aya, anat,
  yehoraz) scores, keyed by annotator then by instance id.
- **`llm_annotations.json`** — the LLM judge's scores for the same
  instances, keyed by instance id.

### `data/raw/`
Original, unmodified input data. Never edit these files by hand.
- **`original_test_dataset.csv`** — the source sarcastic-tweet test set
  used by the interpretation pipeline (from the original SIGN paper's
  repo, header-less, tweet text in the first column).
- **`sarcasm_corpus_v2/`** — input data for the detection/classification
  phase: `GEN-sarc-notsarc.csv` (6,520 posts), `HYP-sarc-notsarc.csv`
  (1,164 posts), `RQ-sarc-notsarc.csv` (1,702 posts) — General Sarcasm,
  Hyperbole, and Rhetorical-Question subsets of Sarcasm Corpus V2 (UC
  Santa Cruz), each with columns `class` (`sarc`/`notsarc`), `id`, `text`.

### `data/processed/`
Cleaned/derived data.
- **`clean_sarcastic_sentences.csv`** — interpretation pipeline: one
  column, `sarcastic_sentence`, deduplicated. Produced by
  `src/preprocessing/clean_dataset.py`.
- **`sarcasm_v2_canonical.csv`** — classification pipeline: the three
  `sarcasm_corpus_v2/` files combined into one table with a global
  `example_id`, `category` (GEN/HYP/RQ), `source_file`, and
  duplicate/label-conflict flags. Produced by
  `src/classification/data/build_canonical_dataset.py`; nothing is
  dropped or deduplicated at this step, only normalized.
- **`sarcasm_v2_audit_report.json`** — data-quality report (class
  balance, duplicates, label conflicts, length distribution) for the
  canonical dataset above. Produced by
  `src/classification/data/audit_dataset.py`.

### `data/splits/`
The one canonical train/dev/test split, reused by every classification
approach. Produced by `src/classification/data/make_splits.py`
(`StratifiedGroupKFold`, grouped by `dup_group_id` so near-duplicate text
can never land in both TRAIN and TEST, stratified by label, seed 42) —
see `PROJECT_SUMMARY.md` §3.1 for the full methodology.
- **`train.csv`** (6,706 rows), **`dev.csv`** (1,340 rows),
  **`test.csv`** (1,340 rows) — each a subset of the canonical dataset's
  columns.
- **`split_assignments.csv`** — `example_id` -> split name, for anyone
  who needs to check which split a given example landed in without
  re-deriving the split.

### `data/llm_cache/`
Per-example disk cache for LLM calls (keyed by request content), so a
re-run after a transient failure or crash doesn't re-spend GPU/API
budget. **Gitignored** — regenerable, not source data.

### `data/model_outputs/`
Interpretation pipeline: one subfolder per experiment run
(`experiment_01`–`experiment_04`), each holding one CSV per generation
model (`gemini_run_0N.csv`, `nvidia_run_0N.csv`, `liquid_run_0N.csv`).
Columns before evaluation: `sarcastic_sentence`, `model_interpretation`;
`classification` (1/2/3, from the LLM judge) added after evaluation.

### `data/manual_scoring/`
Interpretation pipeline: human annotation stage (Apple Numbers files).
- **`random_70_for_manual_scoring.numbers`** — the master sample: 70
  tweets with each model's translations across all 4 prompts.
- **`anat.numbers`, `aya.numbers`, `yehoraz.numbers`** — each of the 3
  team members' independently completed copy, scored 1–3 (feeds the
  Alt-Test and Fleiss' Kappa analysis).

### `data/summaries/`
Interpretation pipeline: aggregated result tables and figures, produced
by `src/postprocessing/*.py` — see that section below for which script
produces which file.

---

## `docs/`

Research documentation — the "why" and "what happened," as opposed to code.

- **`pipeline.md`** — technical, stage-by-stage map of the interpretation
  pipeline: what runs, in what order, and which stages need an API key
  or a model download.
- **`methodology.md`** — *how* the interpretation pipeline's dataset,
  models, prompts, and evaluation were chosen.
- **`results.md`** — *what was found* in the interpretation pipeline:
  automatic-metric tables, Alt-Test outcome, Fleiss' Kappa, significance
  tests, human-vs-LLM-judge agreement, case studies.
- **`alt_test_reference.md`** — what the Alt-Test is, citation for the
  paper it's from, how epsilon was chosen, and where the
  code/data/script live in this repo.
- **`project_structure.md`** — this file.
- **`sign_paper.pdf`** — local copy of the original SIGN paper (ACL 2017)
  that the interpretation pipeline's dataset and starting point are drawn
  from; see `methodology.md`.

---

## `models/`

Trained model checkpoints (currently `EXP-009/best_checkpoint/`, the
fine-tuned DeBERTa-v3-base weights). **Gitignored** — binary artifacts,
not source; kept durable on local disk independently of the VM's
ephemeral storage. Regenerable by re-running
`configs/transformer_deberta_v3_base.json`.

---

## `prompts/`

Every prompt template as a plain `.txt` file, kept out of the Python code
so prompt engineering doesn't require touching scripts. Loaded via
`src/common/prompt_loader.load_prompt(path)` (interpretation pipeline) or
directly by `src/classification/llm/run_llm_classification.py`
(classification pipeline), where `path` is relative to `prompts/`.

### `prompts/generation/` (interpretation pipeline)
- **`generation_prompt_v1.txt`** — plain instruction, no examples.
- **`generation_prompt_v2.txt`** — "translate the true meaning," output
  only the sentence. Most effective prompt with human annotators.
- **`generation_prompt_v3.txt`** — adds formatting/grammar constraints.
  Consistently the worst-performing prompt.
- **`generation_prompt_v4.txt`** — few-shot, with 3 worked examples.
  Current default used by both generation scripts.

### `prompts/evaluation/` (interpretation pipeline)
- **`llm_judge_prompt.txt`** — the 1/2/3 judge prompt used by
  `evaluate_with_llm.py`.
- **`binary_judge_prompt.txt`** — an older/alternative 0/1 binary judge
  prompt, not currently called from any script; kept for reproducibility
  and comparison.
- **`nli_premise_template.txt`** / **`nli_hypothesis_template.txt`** —
  used by `evaluate_with_nli.py`; plain pass-throughs of their single
  placeholder.

### `prompts/classification/` (Stage B)
- **`zero_shot_v1.txt`** — M2's prompt: task definition only, no examples.
- **`few_shot_v1.txt`** — M3's prompt: task definition + a block of
  labeled demonstration examples (selected by
  `src/classification/llm/few_shot_selection.py`).
- **`reasoning_v1.txt`** — M4's prompt: asks for step-by-step reasoning
  before committing to a label.

---

## `results/`

One subfolder per Stage B experiment (`EXP-00N` for DEV runs,
`EXP-00N-TEST` for the corresponding sealed-TEST evaluation), each
holding exactly:
- **`config.json`** — the full experiment configuration used (a copy of
  the `configs/*.json` file plus the resolved `experiment_id`/`eval_split`).
- **`metrics.json`** — accuracy, macro/weighted F1, per-class
  precision/recall/F1, confusion matrix — all computed by the single
  shared implementation in `src/classification/evaluation/metrics.py`.
- **`predictions.csv`** — one row per example: `example_id`, `gold_label`,
  `predicted_label`, optional `confidence`.
- **`compiled_program.json`** *(EXP-008 / EXP-008-TEST only)* — DSPy's
  saved program state for the MIPROv2-optimized method: the winning
  instruction text and the exact few-shot demonstrations selected. See
  `PROJECT_SUMMARY.md` §6.1 for the prompt quoted directly from this file.

Other files:
- **`EXP-001-dev-ref/`** — M1 (TF-IDF+LR) was frozen back in Stage A and
  only ever evaluated on TEST. For the cross-model DEV analysis below,
  the identical frozen config was re-run once with `eval_split=dev` to
  get a comparable DEV prediction file — not a re-tune, a read-only
  reference point.
- **`cross_model_dev_analysis.csv`** / **`cross_model_test_analysis.csv`**
  — one row per example with every method's prediction, correctness, and
  `n_models_correct`, joined against the canonical dataset. Built by an
  ad hoc analysis script; see `PROJECT_SUMMARY.md` §7 for the write-up.

---

## `scripts/`

What remains once Stage B's experiments were all complete and sealed —
the scripts still needed to reproduce a result, not the one-time Azure
GPU VM setup/sync/kernel-verification tooling used to produce them
(removed once the VM was no longer needed).

- **`smoke_test_dspy.py`** — exercises the DSPy/local-Qwen adapter on a
  handful of examples before committing to a full-DEV run.
- **`eval_frozen_checkpoint.py`** — evaluates an already-trained M6
  checkpoint on a given split without retraining (the standalone
  eval-only path `finetune.py` doesn't otherwise provide) — used to
  reproduce M6's TEST result.
- **`run_phase2_test_chain.sh`** — runs every frozen non-M1 config once
  on the sealed TEST split, in sequence; resume-aware (skips any step
  whose result already exists). The exact script used to produce every
  TEST number in `PROJECT_SUMMARY.md`.

---

## `src/`

All Python code, organized by pipeline stage. Every subfolder has an
`__init__.py` (empty, just marks it as a package) and every runnable
script follows the same pattern: a pure function doing the work, plus a
thin `main()` with `argparse` so it can be run as
`python -m src.<subfolder>.<script> [options]`.

### `src/common/` (interpretation pipeline)
Shared helpers used across multiple pipeline stages.
- **`file_utils.py`** — `ensure_parent_dir`, `read_csv_flexible`,
  `save_csv`, `load_all_model_outputs`.
- **`gemini_client.py`** — `get_gemini_model(model_name=None)`.
- **`openrouter_client.py`** — `get_openrouter_client()`.
- **`json_utils.py`** — `extract_json_array(text)`: pulls the first
  `[...]` JSON array out of an LLM response.
- **`prompt_loader.py`** — `load_prompt(relative_path, prompts_dir=None)`.
- **`alt_test.py`** — the reference implementation of the Alt-Test
  (Calderon, Reichart & Dror, 2025). See `docs/alt_test_reference.md`.
- **`nli_utils.py`** — entailment/contradiction label-mapping logic used
  by `evaluate_with_nli.py`.

### `src/preprocessing/`
- **`clean_dataset.py`** — `clean_sarcastic_sentences(input_path,
  output_path)`: dedupes and sorts the raw sarcastic-sentence list. First
  step of the interpretation pipeline.

### `src/generation/`
- **`generate_with_gemini.py`**, **`generate_with_openrouter.py`** — call
  a model row-by-row over the clean sentences, saving after every row so
  a crash/rate limit doesn't lose progress. See `README.md` for usage.

### `src/evaluation/` (interpretation pipeline)
- **`evaluate_with_llm.py`** — the main evaluation method: an LLM judge
  scores every row 1/2/3, in retried batches.
- **`evaluate_with_nli.py`** — an automatic alternative: an NLI model
  checks whether the interpretation entails the original sarcastic
  sentence's true meaning more strongly than it contradicts it.

### `src/postprocessing/` (interpretation pipeline)
Everything that runs after generation + evaluation.
`summarize_classifications.py`, `calculate_text_metrics.py` (BLEU/ROUGE/
PINC/combined score), `create_manual_sample.py`, `run_alt_test.py`,
`summarize_text_metrics.py`, `plot_text_metrics.py`,
`significance_tests.py` (Kruskal-Wallis), `correlation_heatmap.py`,
`linguistic_analysis.py`, `human_llm_agreement.py` (Fleiss' Kappa),
`extract_case_studies.py`. See `docs/results.md` for what each produced.

### `src/tools/`
- **`check_openrouter_limit.py`** — prints current OpenRouter API
  usage/quota; useful before a long run.

### `src/classification/` (Stage B — detection)

- **`run_experiment.py`** — single entry point:
  `python -m src.classification.run_experiment --config configs/<name>.json`.
  Reads a JSON config, dispatches to the right approach module by its
  `approach_family` field.

- **`data/`** — dataset construction, reused once by every approach:
  - `build_canonical_dataset.py` — combines the 3 raw category files into
    one canonical table.
  - `audit_dataset.py` — data-quality report (never silently fixes/drops).
  - `make_splits.py` — the one canonical train/dev/test split (see
    `data/splits/` above).

- **`classical/`** — M1:
  - `tfidf_baseline.py` — TF-IDF vectorizer -> Logistic Regression
    (Linear SVM also supported). Stage A's frozen baseline.

- **`llm/`** — M2/M3/M4 (manual-prompt LLM approaches):
  - `client.py` — client factory, `provider="openrouter"` or
    `"local_hf"`.
  - `local_client.py` — local Hugging Face Transformers inference
    (`Qwen/Qwen3-4B-Instruct-2507`), built for Tesla M60 GPUs (no
    bfloat16/FlashAttention2/vLLM support — plain fp16 `transformers`
    generation).
  - `few_shot_selection.py` — deterministic demo selection from TRAIN
    only, given `(variant, n_shots, seed)`.
  - `run_llm_classification.py` — zero-shot / few-shot / reasoning
    classification: retried on failure, disk-cached, bounded concurrency.
  - `schema.py` — structured-output parsing (every LLM response must
    resolve to exactly one of the two canonical labels).

- **`dspy_pipeline/`** — M5:
  - `local_lm.py` — a `dspy.BaseLM` adapter that runs DSPy programs
    against the local Qwen client, so DSPy uses the exact same model as
    M2–M4 with no external server.
  - `signatures.py` — the DSPy signature for sarcasm classification.
  - `run_dspy.py` — `Predict` / `BootstrapFewShot` / `MIPROv2` variants.
    TRAIN is used for optimization/bootstrapping, DEV is the optimizer's
    validation metric, TEST is only ever touched once per frozen config.

- **`transformer/`** — M6:
  - `finetune.py` — fine-tunes a pretrained encoder
    (`microsoft/deberta-v3-base`) via `transformers.Trainer`, with early
    stopping on DEV Macro F1.

- **`evaluation/`** — shared by every approach:
  - `metrics.py` — `compute_metrics`: the single implementation of every
    metric used for model selection and comparison.
  - `io.py` — persists/loads one experiment's `results/<experiment_id>/`
    artifacts.
  - `error_analysis.py` — cross-model disagreement analysis: merges
    multiple experiments' `predictions.csv` into one wide table, produces
    pairwise disagreement subsets (e.g. "TF-IDF right, Qwen wrong").

---

## `tests/`

Pytest suite (`pytest` from the repo root). Covers both pipelines:
environment-variable validation, prompt loading, CLI `--help` for every
script, text-metric functions, JSON parsing, the Alt-Test algorithm, NLI
label-mapping, mocked API request/response handling (Gemini/OpenRouter/
LLM-judge/quota-check), and the full classification pipeline (dataset
construction, splitting, few-shot selection, metrics, error analysis, the
LLM client/schema with a mocked model, and the local-HF client). Never
calls a real API, never downloads a model, never needs a GPU.

---

## Root files

- **`.env`** — local secrets (`OPENROUTER_API_KEY`, `GEMINI_API_KEY`).
  Never committed (see `.gitignore`).
- **`.gitignore`** — excludes `.env`, Python caches, virtual environments,
  IDE folders, `.DS_Store`, `data/llm_cache/`, `models/`.
- **`README.md`** — quick-start overview and installation.
- **`PROJECT_SUMMARY.md`** — the project's full results, methodology, and
  conclusions for all three parts (interpretation, detection, SIGN
  generalization); the main deliverable.
- **`Sarcasm_Project_Report.docx`** — the full formal write-up,
  submission-ready.
- **`requirements.txt`** — base runtime dependencies (both pipelines).
- **`requirements-classification.txt`** — additional dependencies needed
  only for Stage B (`dspy`, `accelerate`, `sentencepiece`).
- **`requirements-dev.txt`** — + testing dependencies (`pytest`,
  `pytest-mock`).
- **`environment_stage_b.txt`** — the exact pinned package versions
  verified working on the Azure GPU VM (Python/CUDA/driver/`pip freeze`),
  for reproducing the Stage B environment exactly.
