# Project Structure — Full Reference

A complete, folder-by-folder and file-by-file walkthrough of everything in
`sarcasm/`: what it is, what it does, and how it's used. For a quicker
overview and setup instructions, see the main `README.md`. For what was
decided at each meeting, see `docs/meeting_notes_summary.md`.

---

## Top-level layout

```text
sarcasm/
├── config/         # project-wide settings and model-ID constants
├── data/           # all datasets and result files (no code)
├── docs/           # research documentation and meeting slides
├── prompts/        # every prompt template, as plain .txt files
├── src/            # all Python code, one subfolder per pipeline stage
├── .env            # local API keys (not committed to git)
├── .gitignore
├── README.md
└── requirements.txt
```

The project has two tracks:
1. **Interpretation pipeline** (implemented, the bulk of the work so far):
   clean data → generate interpretations with 3 LLMs → judge them with a 4th
   LLM → summarize/analyze results.
2. **Detection fine-tuning** (planned, not yet implemented): fine-tune a BERT
   classifier to detect whether a sentence is sarcastic at all. See
   `docs/finetuning_plan.md`.

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
- **`settings.py`** — defines the `Settings` dataclass (instantiated once as
  `settings`) with:
  - Every data folder path (`raw_data_dir`, `processed_data_dir`,
    `model_outputs_dir`, `manual_scoring_dir`, `summaries_dir`) and
    `prompts_dir`, all computed from `PROJECT_ROOT` so scripts work
    regardless of the current working directory.
  - `openrouter_api_key` / `gemini_api_key` — loaded from `.env` via
    `python-dotenv`.
  - `default_openrouter_model`, `default_judge_model`, `default_gemini_model`
    — pulled from `config/models.py` (not duplicated as separate string
    literals) so there is exactly one place to change a model ID.

---

## `data/`

Datasets and result files only — no code belongs here.

### `data/alt_test/`
The raw data behind the project's Alt-Test result (see
`docs/alt_test_reference.md` and `src/postprocessing/run_alt_test.py`).
- **`humans_annotations.json`** — the 3 human annotators' (aya, anat,
  yehoraz) scores, keyed by annotator then by instance id
  (`"<prompt>_<model>_<sarcastic sentence>"`).
- **`llm_annotations.json`** — the LLM judge's scores for the same
  instances, keyed by instance id.

### `data/raw/`
Original, unmodified input data. Never edit these files by hand.
- **`original_test_dataset.csv`** — the source sarcastic-tweet test set used
  throughout the interpretation pipeline (from the original SIGN paper's
  repo, header-less, tweet text in the first column).
- **`sarcasm_corpus_v2/`** — input data staged for the *upcoming* detection
  fine-tuning phase (not yet used by any script):
  - `GEN-sarc-notsarc.csv` (6,520 posts), `HYP-sarc-notsarc.csv` (1,164
    posts), `RQ-sarc-notsarc.csv` (1,702 posts) — General Sarcasm,
    Hyperbole, and Rhetorical-Question subsets of Sarcasm Corpus V2 (UC
    Santa Cruz), each with columns `class` (`sarc`/`notsarc`), `id`, `text`.
    See `docs/finetuning_plan.md`.

### `data/processed/`
Cleaned data derived from `raw/`, produced by
`src/preprocessing/clean_dataset.py`.
- **`clean_sarcastic_sentences.csv`** — one column, `sarcastic_sentence`:
  deduplicated sarcastic sentences extracted from
  `raw/original_test_dataset.csv`. This is the input to both generation
  scripts.

### `data/model_outputs/`
One subfolder per experiment run, each holding one CSV per model.
- **`experiment_01/` … `experiment_04/`** — `gemini_run_0N.csv`,
  `nvidia_run_0N.csv`, `liquid_run_0N.csv`. Experiment number corresponds to
  which generation prompt version was used (e.g. `experiment_04` used
  `generation_prompt_v4.txt`, the few-shot prompt).
  - Columns before evaluation: `sarcastic_sentence`, `model_interpretation`.
  - Column added after evaluation: `classification` (1/2/3, from the LLM
    judge).

### `data/manual_scoring/`
Human annotation stage. Apple Numbers files (open in Numbers, Excel, or
Google Sheets).
- **`random_70_for_manual_scoring.numbers`** — the master sample: 70 tweets
  with each model's translations across all 4 prompts, and empty score
  columns.
- **`anat.numbers`, `aya.numbers`, `yehoraz.numbers`** — each of the 3 team
  members' independently completed copy of the master sample, scored 1-3
  (see `docs/meeting_notes_summary.md`, 2026-07-16 meeting, for how these
  feed into the Alt-Test and Fleiss' Kappa analysis).

### `data/summaries/`
Aggregated result tables, produced by the `src/postprocessing/` scripts.
- **`classification_summary.csv`** — one row per experiment/model, with
  average/median score and counts of each score (1/2/3). Produced by
  `summarize_classifications.py`.
- **`text_metrics_nvidia_run_01.csv`** — BLEU/ROUGE/PINC/Combined averages
  for one model-output file. Produced by `calculate_text_metrics.py` (one
  file per run; re-run for other models/experiments as needed).
- **`text_metrics_summary.csv`** — the same metrics for every prompt/model
  combination at once. Produced by `summarize_text_metrics.py`.
- **`perfect_agreement_cases.csv`** / **`discrepancy_cases_for_discussion.csv`**
  — case studies where the human and LLM-judge scores agree / disagree
  most. Produced by `extract_case_studies.py`.
- **`figures/`** — every plot produced by the postprocessing scripts
  (`plot_text_metrics.py`, `correlation_heatmap.py`, `linguistic_analysis.py`,
  `human_llm_agreement.py`): NLP-metric distributions, the Spearman
  correlation heatmap, structural/linguistic comparisons, and the
  human-vs-ChatGPT comparison plots and confusion matrix.

---

## `docs/`

Research documentation — the "why" and "what happened," as opposed to code.

- **`meeting_notes_summary.md`** — full, meeting-by-meeting record of all
  four project meetings (models tested, prompts, metrics, Alt-Test, error
  analysis, decisions). Start here instead of opening the slide decks.
- **`alt_test_reference.md`** — what the Alt-Test is, citation for the
  paper it's from, how epsilon was chosen, and where the
  code/data/script live in this repo.
- **`finetuning_plan.md`** — the plan for the next phase (BERT-based
  sarcasm detection fine-tuning), based on the last 2 slides of the
  2026-07-16 meeting.
- **`methodology_from_meetings.md`** — narrative summary of *how* the
  dataset was prepared, models selected, outputs generated, evaluation
  performed, and prompt sensitivity tested.
- **`results_from_meetings.md`** — narrative summary of *results*: model
  comparisons, prompt sensitivity findings, good/bad interpretation
  examples.
- **`pipeline.md`** — step-by-step walkthrough of the interpretation
  pipeline (complements the "Full Pipeline" diagram in the main README).
- **`meeting_slides/`** — the original `.pptx` decks, one per meeting, named
  `meeting_YYYY-MM-DD_sarcasm_sign.pptx`:
  - `meeting_2026-05-28_sarcasm_sign.pptx` — model survey & automatic
    metrics.
  - `meeting_2026-06-16_sarcasm_sign.pptx` — judge architecture & prompt
    sensitivity.
  - `meeting_2026-07-09_sarcasm_sign.pptx` — Alt-Test draft.
  - `meeting_2026-07-16_sarcasm_sign.pptx` — Alt-Test finalized, full error
    analysis, and the fine-tuning plan (most complete deck; supersedes the
    07-09 draft).

---

## `prompts/`

Every prompt template as a plain `.txt` file, kept out of the Python code so
prompt engineering doesn't require touching scripts. All generation prompts
use `{sarcastic_sentence}` as their placeholder (consistent across all four
versions); evaluation prompts use `{sarcastic_sentence}` /
`{model_interpretation}` as needed. Loaded via
`src/common/prompt_loader.load_prompt(path)`, where `path` is relative to
`prompts/` (e.g. `"generation/generation_prompt_v4.txt"`).

### `prompts/generation/`
- **`generation_prompt_v1.txt`** — Prompt 1: plain instruction, no examples.
- **`generation_prompt_v2.txt`** — Prompt 2: "translate the true meaning,"
  output only the sentence. The most effective prompt with human
  annotators (see meeting notes).
- **`generation_prompt_v3.txt`** — Prompt 3: adds formatting/grammar
  constraints. Consistently the worst-performing prompt.
- **`generation_prompt_v4.txt`** — Prompt 4: few-shot, with 3 worked
  examples baked into the prompt. Current default used by both generation
  scripts.

### `prompts/evaluation/`
- **`llm_judge_prompt.txt`** — the 1/2/3 judge prompt used by
  `evaluate_with_llm.py` (the main evaluation method).
- **`binary_judge_prompt.txt`** — an older/alternative 0/1 binary judge
  prompt, not currently called from any script; kept for reproducibility
  and comparison.
- **`nli_premise_template.txt`** / **`nli_hypothesis_template.txt`** — used
  by `evaluate_with_nli.py`. Both are currently a plain pass-through of
  their single placeholder (`{sarcastic_sentence}` /
  `{model_interpretation}`), since the NLI model just needs the raw
  sentence and interpretation as premise/hypothesis.

---

## `src/`

All Python code, organized by pipeline stage. Every subfolder has an
`__init__.py` (empty, just marks it as a package) and every runnable script
follows the same pattern: a pure function doing the work, plus a thin
`main()` with `argparse` so it can be run as
`python -m src.<subfolder>.<script> [options]`.

### `src/common/`
Shared helpers used across multiple pipeline stages.
- **`file_utils.py`**
  - `ensure_parent_dir(path)` — creates a file's parent directory if
    missing.
  - `read_csv_flexible(path, expected_columns)` — reads a CSV; if the
    expected columns aren't present as a header, re-reads it headerless and
    assigns `expected_columns` (handles the original dataset's header-less
    format).
  - `save_csv(df, path)` — saves as UTF-8-SIG (so Excel/Numbers open it
    correctly), creating the parent directory if needed.
  - `load_all_model_outputs(model_outputs_dir)` — loads every classified CSV
    under `data/model_outputs/` into one DataFrame, adding `prompt` and
    `model` columns. Used by the statistical/linguistic analysis scripts
    below, which need the full labeled dataset rather than one file.
- **`gemini_client.py`** — `get_gemini_model(model_name=None)`: configures
  `google.generativeai` with the API key from settings and returns a
  `GenerativeModel` (defaults to `settings.default_gemini_model`). Raises a
  clear error if `GEMINI_API_KEY` is missing.
- **`openrouter_client.py`** — `get_openrouter_client()`: returns an
  OpenAI-SDK-compatible client pointed at OpenRouter's API. Raises a clear
  error if `OPENROUTER_API_KEY` is missing.
- **`json_utils.py`** — `extract_json_array(text)`: pulls the first `[...]`
  JSON array out of an LLM response, stripping any Markdown code fences
  first. Used to parse the judge's batch score responses.
- **`prompt_loader.py`** — `load_prompt(relative_path, prompts_dir=None)`:
  reads a prompt file from `prompts/` (or a custom directory), raising
  `FileNotFoundError` with the resolved path if it's missing.
- **`alt_test.py`** — `alt_test(llm_annotations, humans_annotations,
  scoring_function="accuracy", epsilon=0.2, ...)`: the reference
  implementation of the Alt-Test (Calderon, Reichart & Dror, 2025). Leaves
  out one human annotator at a time and checks whether the LLM or the
  excluded human better matches the *remaining* annotators; returns
  `(winning_rate, advantage_prob)`. Also exposes the scoring helpers
  `accuracy`, `neg_rmse`, `sim`. See `docs/alt_test_reference.md`.

### `src/preprocessing/`
- **`clean_dataset.py`** — `clean_sarcastic_sentences(input_path,
  output_path)`: reads the raw, header-less test dataset, takes the first
  column, splits off anything after the first comma (defensive against
  stray extra columns), strips whitespace, deduplicates, and sorts. Saves
  one column: `sarcastic_sentence`. This is the first step of the
  interpretation pipeline.
  ```bash
  python -m src.preprocessing.clean_dataset --input data/raw/original_test_dataset.csv --output data/processed/clean_sarcastic_sentences.csv
  ```

### `src/generation/`
Both scripts share the same structure: read the clean sentences, load a
prompt, call a model row-by-row, and save after *every* row (so a crash or
rate limit doesn't lose progress — re-running resumes from unfinished rows).
- **`generate_with_gemini.py`** — `generate_interpretations(input_path,
  output_path, prompt_name="generation/generation_prompt_v4.txt",
  start_row=0, end_row=None, sleep_seconds=2.0)`. Calls Gemini with all 4
  harm-category safety filters set to `BLOCK_NONE` (sarcastic tweets often
  contain mild profanity; Gemini blocks them by default instead of
  translating — see the safety-refusal case study in the meeting notes).
  This doesn't eliminate every refusal. On any exception, writes `"ERROR"`
  for that row and continues (re-run to retry failed rows).
  ```bash
  python -m src.generation.generate_with_gemini --input data/processed/clean_sarcastic_sentences.csv --output data/model_outputs/experiment_new/gemini.csv --prompt generation/generation_prompt_v2.txt
  ```
- **`generate_with_openrouter.py`** — same shape, generalized to any
  OpenRouter-hosted model (Nvidia, Liquid, or anything else available on
  OpenRouter) via `--model`.
  ```bash
  python -m src.generation.generate_with_openrouter --input data/processed/clean_sarcastic_sentences.csv --output data/model_outputs/experiment_new/nvidia.csv --model nvidia/nemotron-nano-9b-v2:free --prompt generation/generation_prompt_v4.txt
  ```
  Both scripts accept `--prompt <path relative to prompts/>` to pick which
  generation prompt version to use; it defaults to Prompt 4 (few-shot).

### `src/evaluation/`
- **`evaluate_with_llm.py`** — the main evaluation method: an LLM judge
  scores every unclassified row 1/2/3.
  - `build_batch_examples(batch_df)` — formats a batch of rows into the
    numbered "Example N / Sarcastic sentence / Model interpretation" text
    block sent to the judge.
  - `classify_batch(batch_df, model_id, max_retries=5, wait_seconds=15)` —
    sends one batch to the judge, parses the JSON array of scores back out,
    validates the count and that every score is in `{1,2,3}`, retrying with
    a wait on failure (rate limits, malformed JSON, etc.).
  - `evaluate_file(input_path, output_path=None, model_id=None,
    batch_size=10)` — finds rows without a `classification` value yet,
    evaluates them in batches, saving after every batch.
  - `evaluate_directory(outputs_dir, ...)` — runs `evaluate_file` over every
    CSV under a directory tree (i.e. all experiments at once).
  ```bash
  python -m src.evaluation.evaluate_with_llm --directory data/model_outputs --model openai/gpt-oss-20b:free --batch-size 10
  ```
- **`evaluate_with_nli.py`** — an alternative, fully automatic evaluation
  method using an NLI (natural language inference) model instead of an LLM
  judge: treats the sarcastic sentence as the *premise* and the model's
  interpretation as the *hypothesis*, and checks whether the model predicts
  entailment more strongly than contradiction. Adds an `nli_success` column
  (1/0). Defaults to `config.models.JUDGE_MODELS["nli"]`
  (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`), overridable via `--model`.
  Runs on GPU if available, otherwise CPU.
  ```bash
  python -m src.evaluation.evaluate_with_nli --input data/model_outputs/experiment_04/nvidia_run_04.csv --output data/model_outputs/experiment_04/nvidia_nli.csv
  ```

### `src/postprocessing/`
Everything that runs after generation + evaluation are done.
- **`summarize_classifications.py`** — `infer_model_name(csv_path)` turns a
  filename like `nvidia_run_04.csv` into the readable `"nvidia run 04"`.
  `summarize_outputs(outputs_dir, output_file)` walks every CSV under
  `data/model_outputs/`, skips files with no `classification` column or no
  classified rows, and builds one summary row per file: experiment name,
  model name, average/median score, and counts of each score.
  ```bash
  python -m src.postprocessing.summarize_classifications --outputs-dir data/model_outputs --output data/summaries/classification_summary.csv
  ```
- **`calculate_text_metrics.py`** — computes BLEU (via `nltk`), ROUGE-1/2
  (via `rouge-score`, with a manual n-gram-recall fallback if that package
  isn't installed), PINC (fraction of interpretation words not present in
  the original — pure Python, no dependency), and the combined score
  `PINC * sigmoid(BLEU)`, then averages across all rows in one file.
  ```bash
  python -m src.postprocessing.calculate_text_metrics --input data/model_outputs/experiment_04/nvidia_run_04.csv --output data/summaries/text_metrics_nvidia_run_01.csv
  ```
- **`create_manual_sample.py`** — `find_one_csv(folder, keyword)` locates
  exactly one CSV matching a model-name keyword inside an experiment
  folder (raises if zero or more than one match). `create_sample(
  outputs_dir, output_file, sample_size=70, seed=42)` picks `sample_size`
  random row indices from the first experiment's Gemini file, then for each
  index picks a *random experiment* and pulls that experiment's Gemini/
  Nvidia/Liquid interpretations for that same row index, building one
  side-by-side comparison table with empty score columns. This is what
  produced `random_70_for_manual_scoring.numbers` (originally as a `.csv`,
  then opened and saved in Numbers for annotation).
  ```bash
  python -m src.postprocessing.create_manual_sample --outputs-dir data/model_outputs --output data/manual_scoring/random_70_for_manual_scoring.csv --sample-size 70 --seed 42
  ```
- **`run_alt_test.py`** — loads `data/alt_test/humans_annotations.json` and
  `llm_annotations.json`, calls `src.common.alt_test.alt_test(...)`, and
  prints the Winning Rate and Advantage Probability. Reproduces the
  project's documented result exactly (Winning Rate 0.67, Advantage
  Probability 0.77, 3 instances dropped for having fewer than 2 annotators).
  ```bash
  python -m src.postprocessing.run_alt_test
  ```
- **`summarize_text_metrics.py`** — runs `calculate_text_metrics.calculate_metrics`
  (reused, not reimplemented) over every file under `data/model_outputs/`,
  producing one BLEU/ROUGE/PINC row per prompt/model combination.
  ```bash
  python -m src.postprocessing.summarize_text_metrics
  ```
- **`plot_text_metrics.py`** — boxplots of the above summary, by model and
  by prompt. Saves to `data/summaries/figures/`.
  ```bash
  python -m src.postprocessing.plot_text_metrics
  ```
- **`significance_tests.py`** — Kruskal-Wallis tests for whether prompt
  choice and model choice significantly affect the classification score.
  Reproduces the documented result exactly (prompt p=9.45e-34, model
  p=5.13e-72).
  ```bash
  python -m src.postprocessing.significance_tests
  ```
- **`correlation_heatmap.py`** — computes structural metrics (sentence
  length difference, lexical overlap ratio) and plots their Spearman
  correlation with the quality score. Exposes `add_structural_metrics(df)`,
  reused by `linguistic_analysis.py`.
  ```bash
  python -m src.postprocessing.correlation_heatmap
  ```
- **`linguistic_analysis.py`** — 3 plots: source vs. translated sentence
  length per model, lexical overlap across prompts/models, and overlap vs.
  quality score. Reuses `add_structural_metrics` from `correlation_heatmap.py`.
  ```bash
  python -m src.postprocessing.linguistic_analysis
  ```
- **`human_llm_agreement.py`** — joins `data/alt_test/humans_annotations.json`
  and `llm_annotations.json` with the interpretation text in
  `data/model_outputs/` (no separate copy of this data is stored anywhere).
  Computes Fleiss' Kappa among the 3 human annotators (reproduces 0.282
  exactly), and plots model/prompt score comparisons, a human-vs-ChatGPT
  scatter plot, and a confusion matrix (reproduces the documented 47
  misclassified score-2 instances and the 67.1%/31.4% agreement rates on
  Liquid/Nvidia exactly). Exposes `load_human_scores`, `load_llm_scores`,
  `attach_interpretations`, and `build_comparison_table`, reused by
  `extract_case_studies.py`.
  ```bash
  python -m src.postprocessing.human_llm_agreement
  ```
- **`extract_case_studies.py`** — using the same comparison table as
  `human_llm_agreement.py`, saves every case where the human and ChatGPT
  scores agree (`perfect_agreement_cases.csv`, reproduces 104/210 = 49.5%
  exactly) and the top 10 cases where they disagree most
  (`discrepancy_cases_for_discussion.csv`) for qualitative review.
  ```bash
  python -m src.postprocessing.extract_case_studies
  ```

### `src/tools/`
Utility scripts that support the pipeline but aren't part of the research
pipeline itself.
- **`check_openrouter_limit.py`** — calls OpenRouter's `/api/v1/key`
  endpoint with the configured API key and pretty-prints the current
  usage/limit as JSON. Useful before a long generation/evaluation run to
  confirm there's enough quota left.
  ```bash
  python -m src.tools.check_openrouter_limit
  ```

---

## Root files

- **`.env`** — local secrets (`OPENROUTER_API_KEY`, `GEMINI_API_KEY`).
  Never committed (see `.gitignore`).
- **`.gitignore`** — excludes `.env`, Python caches, virtual environments,
  IDE folders, and `.DS_Store`.
- **`README.md`** — quick-start overview: project goal, pipeline diagram,
  folder structure, installation, and the recommended run order.
- **`requirements.txt`** — pinned/bounded dependency versions. A few
  ranges are deliberately capped rather than left open-ended, to avoid
  known breakage (see comments in the file's git history / ask if a version
  bump seems needed): `numpy<2.0.0` (binary compatibility with the pinned
  `torch`), `transformers<5.0.0` (the 5.x line requires `torch>=2.4`, which
  isn't available for every platform this project runs on), and
  `cryptography` pinned to a known-stable range (a newer release had a
  broken wheel on at least one team member's machine).
