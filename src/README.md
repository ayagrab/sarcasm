# `src/` — Code Overview

All Python code for the project, organized by pipeline stage. Every
subfolder is an importable package (`__init__.py`, empty) and every
runnable script follows the same pattern: a pure function doing the work,
plus a thin `main()` with `argparse`, run as:

```bash
python -m src.<subfolder>.<script> [options]
```

(always from the repository root, so the `config`/`src` imports resolve).

For exact commands and arguments, see the root
[`README.md`](../README.md) and [`docs/pipeline.md`](../docs/pipeline.md).
For the full project structure (including `data/`, `docs/`, `prompts/`),
see [`docs/project_structure.md`](../docs/project_structure.md) — this
file covers `src/` only.

---

## `common/`

Shared helpers used across multiple pipeline stages. Nothing here is
runnable on its own.

- **`file_utils.py`** — CSV and filesystem helpers:
  - `ensure_parent_dir(path)` — creates a file's parent directory if missing.
  - `read_csv_flexible(path, expected_columns)` — reads a CSV; if none of
    `expected_columns` appear in the parsed header, re-reads it headerless
    (handles the original SIGN-paper dataset, which has no header row).
  - `save_csv(df, path)` — saves as UTF-8-SIG (so Excel/Numbers open it
    correctly), creating the parent directory if needed.
  - `load_all_model_outputs(model_outputs_dir)` — loads every classified
    CSV under `data/model_outputs/` into one DataFrame with `prompt`/`model`
    columns added; used by the statistical/linguistic analysis scripts.
- **`gemini_client.py`** — `get_gemini_model(model_name=None)`: configures
  `google.generativeai` with the API key from `config.settings` and returns
  a `GenerativeModel`. Raises a clear `RuntimeError` if `GEMINI_API_KEY` is
  missing.
- **`openrouter_client.py`** — `get_openrouter_client()`: returns an
  OpenAI-SDK-compatible client pointed at OpenRouter's API. Raises a clear
  `RuntimeError` if `OPENROUTER_API_KEY` is missing.
- **`json_utils.py`** — `extract_json_array(text)`: pulls the first `[...]`
  JSON array out of an LLM response, stripping Markdown code fences first.
  Used to parse the judge's batch score responses.
- **`prompt_loader.py`** — `load_prompt(relative_path, prompts_dir=None)`:
  reads a prompt file from `prompts/` (path relative to it), raising
  `FileNotFoundError` with the resolved path if missing.
- **`alt_test.py`** — the Alt-Test reference implementation (Calderon,
  Reichart & Dror, 2025 — see `docs/alt_test_reference.md`):
  `alt_test(llm_annotations, humans_annotations, scoring_function, epsilon, ...)`
  returns `(winning_rate, advantage_prob)`. Also exposes the scoring
  helpers `accuracy`, `neg_rmse`, `sim`.
- **`nli_utils.py`** — `classify_entailment(probabilities, id2label, fallback=...)`:
  the entailment-vs-contradiction decision logic for NLI evaluation,
  extracted out of `evaluation/evaluate_with_nli.py` so it's unit-testable
  without downloading any model. Never assumes a fixed numeric label order —
  resolves label meaning from the model's own `id2label` config, falling
  back to an explicit, documented mapping only if the config uses generic
  `LABEL_0`-style placeholders.

## `preprocessing/`

- **`clean_dataset.py`** — `clean_sarcastic_sentences(input_path, output_path)`:
  reads the raw, header-less test dataset
  (`data/raw/original_test_dataset.csv`), extracts the sarcastic sentence
  from the first column, deduplicates, and saves one column,
  `sarcastic_sentence`, to `data/processed/clean_sarcastic_sentences.csv`.
  First step of the interpretation pipeline; needs no API key.

## `generation/`

Both scripts share the same shape: read the clean sentences, load a
generation prompt, call a model row-by-row, and save after *every* row (so
a crash or rate limit doesn't lose progress — re-running resumes
unfinished rows automatically).

- **`generate_with_gemini.py`** — `generate_interpretations(input_path,
  output_path, prompt_name="generation/generation_prompt_v4.txt",
  start_row=0, end_row=None, sleep_seconds=2.0)`. Calls Gemini with all 4
  harm-category safety filters relaxed (`BLOCK_NONE`) since sarcastic
  tweets often contain mild profanity that Gemini otherwise blocks outright
  instead of translating — this doesn't eliminate every refusal (see
  `docs/results.md`). Needs `GEMINI_API_KEY`.
- **`generate_with_openrouter.py`** — same shape, generalized to any
  OpenRouter-hosted model (Nvidia, Liquid, or others) via `--model`. Needs
  `OPENROUTER_API_KEY`.

Both accept `--prompt <path relative to prompts/>` to pick which generation
prompt version to use (default: Prompt 4, few-shot).

## `evaluation/`

- **`evaluate_with_llm.py`** — the main evaluation method: an independent
  LLM judge scores every unclassified row 1/2/3.
  - `build_batch_examples(batch_df)` — formats a batch into the numbered
    "Example N / Sarcastic sentence / Model interpretation" text sent to
    the judge.
  - `classify_batch(batch_df, model_id, max_retries=5, wait_seconds=15)` —
    sends one batch, parses the judge's JSON score array, validates the
    count and that every score is in `{1,2,3}`, retrying on failure.
  - `evaluate_file(input_path, ...)` — evaluates only unclassified rows in
    one CSV, saving after every batch.
  - `evaluate_directory(outputs_dir, ...)` — runs `evaluate_file` over
    every CSV under a directory tree. Needs `OPENROUTER_API_KEY` (the judge
    model is hosted via OpenRouter).
- **`evaluate_with_nli.py`** — an alternative, fully automatic evaluation
  using an NLI model: treats the sarcastic sentence as the *premise* and
  the interpretation as the *hypothesis*, and checks whether the model
  predicts entailment more strongly than contradiction (via
  `common.nli_utils.classify_entailment`). Adds an `nli_success` column
  (1/0). Defaults to `config.models.JUDGE_MODELS["nli"]`
  (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`), overridable via `--model`.
  Needs no API key, but downloads the model from Hugging Face on first run;
  runs on GPU if available, otherwise CPU.

## `postprocessing/`

Everything that runs after generation + evaluation. None of these need an
API key or a model download — they only read files already in `data/`.

- **`summarize_classifications.py`** — one summary row per
  experiment/model: average/median score and counts of each score (1/2/3),
  from `data/model_outputs/`.
- **`calculate_text_metrics.py`** — BLEU (`nltk`), ROUGE-1/2
  (`rouge-score`, with a manual n-gram-recall fallback if that package
  isn't installed), PINC (pure Python), and the combined score
  `PINC * sigmoid(BLEU)`, for one model-output file.
- **`summarize_text_metrics.py`** — runs `calculate_text_metrics`'s
  `calculate_metrics` (reused, not reimplemented) over every file under
  `data/model_outputs/` at once.
- **`plot_text_metrics.py`** — boxplots of the above summary, by model and
  by prompt. Saves to `data/summaries/figures/`.
- **`create_manual_sample.py`** — `find_one_csv(folder, keyword)` locates
  exactly one CSV matching a model-name keyword (raises if zero or more
  than one match). `create_sample(...)` builds a random side-by-side
  comparison sample across experiments/models with empty score columns for
  human annotation.
- **`significance_tests.py`** — Kruskal-Wallis tests for whether prompt
  choice and model choice significantly affect the classification score.
- **`correlation_heatmap.py`** — computes structural metrics (sentence
  length difference, lexical overlap ratio) via `add_structural_metrics(df)`
  and plots their Spearman correlation with the quality score. Reused by
  `linguistic_analysis.py`.
- **`linguistic_analysis.py`** — 3 plots: source vs. translated sentence
  length per model, lexical overlap across prompts/models, and overlap vs.
  quality score.
- **`run_alt_test.py`** — loads `data/alt_test/humans_annotations.json` and
  `llm_annotations.json`, calls `common.alt_test.alt_test(...)`, and prints
  the Winning Rate / Advantage Probability.
- **`human_llm_agreement.py`** — joins the Alt-Test annotation JSON files
  with the interpretation text in `data/model_outputs/` (no separate copy
  of this data is stored anywhere). Computes Fleiss' Kappa among the human
  annotators and plots model/prompt score comparisons, a human-vs-judge
  scatter plot, and a confusion matrix. Exposes `load_human_scores`,
  `load_llm_scores`, `attach_interpretations`, `build_comparison_table`,
  reused by `extract_case_studies.py`.
- **`extract_case_studies.py`** — using the same comparison table as
  `human_llm_agreement.py`, saves every case where the human and judge
  scores agree (`perfect_agreement_cases.csv`) and the top N cases where
  they disagree most (`discrepancy_cases_for_discussion.csv`).

## `tools/`

Utility scripts that support the pipeline but aren't part of the research
pipeline itself.

- **`check_openrouter_limit.py`** — calls OpenRouter's `/api/v1/key`
  endpoint and pretty-prints the current usage/limit as JSON. Useful before
  a long generation/evaluation run to confirm there's enough quota left.
  Needs `OPENROUTER_API_KEY`.
