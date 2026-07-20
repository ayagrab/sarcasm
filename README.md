# Sarcasm Interpretation Benchmark

This project evaluates how well different language models interpret sarcastic tweets as clear, sincere, non-sarcastic statements.

The project is based on the task of **sarcasm interpretation**: converting a sarcastic sentence into a clear statement that preserves the intended meaning and sentiment.

```text
Original sarcastic tweet
        ↓
LLM generates a sincere interpretation
        ↓
The interpretation is evaluated
        ↓
Results are summarized and compared across models/prompts
```

---

## Project Goal

The goal is to compare different LLMs on sarcasm interpretation.

For each sarcastic tweet, a model should output one natural sentence that expresses the author’s real meaning.

Example:

```text
Sarcastic tweet:
4 more assignments wooo college is so fun

Possible interpretation:
I am frustrated about having four more assignments.
```

The project supports:

- cleaning the original dataset
- generating interpretations with Gemini
- generating interpretations with OpenRouter models such as Nvidia and Liquid
- evaluating outputs with an LLM judge
- optionally evaluating outputs with an NLI model
- calculating BLEU, ROUGE, PINC and combined metrics
- creating a random sample for manual scoring
- summarizing final results

---

## Full Pipeline

```text
data/raw/original_test_dataset.csv
        ↓
src/preprocessing/clean_dataset.py
        ↓
clean dataset
        ↓
src/generation/generate_with_gemini.py
src/generation/generate_with_openrouter.py
        ↓
data/model_outputs/experiment_XX/*.csv
        ↓
src/evaluation/evaluate_with_llm.py
src/evaluation/evaluate_with_nli.py  optional
        ↓
model output CSV files with classification scores
        ↓
src/postprocessing/summarize_classifications.py
src/postprocessing/calculate_text_metrics.py
src/postprocessing/create_manual_sample.py
        ↓
data/summaries/
data/manual_scoring/
```

---

## Project Structure

```text
sarcasm/
│
├── config/
│   ├── __init__.py
│   ├── models.py
│   └── settings.py
│
├── data/
│   ├── alt_test/
│   ├── manual_scoring/
│   ├── model_outputs/
│   │   ├── experiment_01/
│   │   ├── experiment_02/
│   │   ├── experiment_03/
│   │   └── experiment_04/
│   ├── processed/
│   ├── raw/
│   │   └── sarcasm_corpus_v2/    # input staged for the next (fine-tuning) phase
│   └── summaries/
│
├── docs/
│   ├── meeting_slides/
│   ├── alt_test_reference.md
│   ├── finetuning_plan.md
│   ├── meeting_notes_summary.md
│   ├── methodology_from_meetings.md
│   ├── pipeline.md
│   ├── project_structure.md
│   └── results_from_meetings.md
│
├── prompts/
│   ├── evaluation/
│   │   ├── binary_judge_prompt.txt
│   │   ├── llm_judge_prompt.txt
│   │   ├── nli_hypothesis_template.txt
│   │   └── nli_premise_template.txt
│   └── generation/
│       ├── generation_prompt_v1.txt
│       ├── generation_prompt_v2.txt
│       ├── generation_prompt_v3.txt
│       └── generation_prompt_v4.txt
│
├── src/
│   ├── common/
│   ├── evaluation/
│   ├── generation/
│   ├── postprocessing/
│   ├── preprocessing/
│   └── tools/
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

For a detailed explanation of every folder and every file, see
**[`docs/project_structure.md`](docs/project_structure.md)**.

---

## Root Files

### `.env`

Local file for API keys.

This file should exist on your computer, but should **not** be uploaded to GitHub.

Example:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
GEMINI_API_KEY=your_gemini_key_here
```

### `.gitignore`

Prevents private or unnecessary files from being uploaded to GitHub.

Important entries should include:

```text
.env
__pycache__/
*.pyc
.venv/
venv/
.idea/
.vscode/
```

### `requirements.txt`

Contains the Python packages required to run the project.

Install them with:

```bash
pip install -r requirements.txt
```

### `README.md`

This file. It explains the project structure, purpose, and how to run each stage.

---

## `config/`

The `config` folder stores project-wide configuration.

### `config/settings.py`

Central project settings.

It defines:

- project root path
- data folders
- prompts folder
- API keys loaded from `.env`
- default model names (pulled from `config/models.py`, not duplicated here)

Other scripts import settings from here instead of hardcoding paths or keys.

Example:

```python
from config.settings import settings

print(settings.model_outputs_dir)
print(settings.openrouter_api_key)
```

### `config/models.py`

Stores model identifiers used in the project.

Example:

```python
GENERATION_MODELS = {
    "gemini": "gemini-2.5-flash-lite",
    "nvidia": "nvidia/nemotron-nano-9b-v2:free",
    "liquid": "liquid/lfm-2.5-1.2b-thinking:free",
}
```

This makes it easier to reuse model names without copying long strings across the code.

### `config/__init__.py`

Marks `config/` as a Python package. Usually this file can stay empty.

---

## `data/`

The `data` folder stores datasets and result files. No Python code should be placed here.

### `data/alt_test/`

The raw data behind the Alt-Test result reported in the meeting notes
(Winning Rate 0.67, Advantage Probability 0.77) -- see
`docs/alt_test_reference.md`.

```text
humans_annotations.json   # the 3 human annotators' scores, keyed by annotator then instance
llm_annotations.json      # the LLM judge's scores for the same instances
```

Run `python -m src.postprocessing.run_alt_test` to reproduce the result.

### `data/raw/`

Original input data. Do not manually edit files in `raw/`.

```text
original_test_dataset.csv     # source test set for the interpretation pipeline
sarcasm_corpus_v2/            # GEN/HYP/RQ files staged for the fine-tuning phase
                               # (not yet used by any script -- see docs/finetuning_plan.md)
```

### `data/model_outputs/`

Contains model-generated interpretations.

Each experiment folder corresponds to a prompt version or experiment run:

```text
data/model_outputs/
├── experiment_01/
├── experiment_02/
├── experiment_03/
└── experiment_04/
```

Each experiment folder should contain CSV files for the models used in that experiment, for example:

```text
gemini.csv
nvidia.csv
liquid.csv
```

Expected columns:

```text
sarcastic_sentence
model_interpretation
classification
```

`classification` is added later by the evaluation stage.

### `data/manual_scoring/`

Contains files used for the human annotation stage (see `docs/results_from_meetings.md`
for the Alt-Test / Fleiss' Kappa analysis built on top of these scores).

Current files (Apple Numbers spreadsheets, viewable in Numbers, Excel, or Google Sheets):

```text
random_70_for_manual_scoring.numbers   # the master sample: 70 tweets with each
                                        # model's translations and empty score columns
anat.numbers                           # annotator "anat"'s completed scores
aya.numbers                            # annotator "aya"'s completed scores
yehoraz.numbers                        # annotator "yehoraz"'s completed scores
```

Each of the three per-annotator files is an independently scored copy of the master
sample, following the "3 team members" methodology described in the meeting slides.

### `data/summaries/`

Contains summary tables, final metrics, and figures.

```text
classification_summary.csv               # per-experiment/model score summary
text_metrics_nvidia_run_01.csv            # BLEU/ROUGE/PINC for one file
text_metrics_summary.csv                  # BLEU/ROUGE/PINC for every prompt/model
perfect_agreement_cases.csv               # human/LLM-judge score matches
discrepancy_cases_for_discussion.csv      # human/LLM-judge score gaps (top 10)
figures/                                  # every plot from the postprocessing scripts
```

Use this folder for aggregated result files.

---

## `docs/`

The `docs` folder explains the research and project decisions.

### `docs/meeting_notes_summary.md`

A full, meeting-by-meeting record of everything covered across all four
meeting decks (models tested, prompts, metrics, Alt-Test, error analysis,
decisions) -- written so you don't need to reopen the `.pptx` files.

### `docs/alt_test_reference.md`

What the Alt-Test is, the paper it's from, how epsilon was chosen, and where
the code/data/script for it live in this repo.

### `docs/finetuning_plan.md`

The plan for the next phase of the project: a BERT-based sarcasm-detection
classifier, based on the last two slides of the 2026-07-16 meeting. Planning
document only -- no training code yet.

### `docs/project_structure.md`

A detailed, folder-by-folder and file-by-file explanation of the entire
repository -- what each file does and how. This README covers the same
ground at a higher level; that doc is the exhaustive reference.

### `docs/meeting_slides/`

Contains the original meeting presentations, named `meeting_YYYY-MM-DD_sarcasm_sign.pptx`
by the date they were presented:

```text
meeting_2026-05-28_sarcasm_sign.pptx   # Meeting 1: model evaluation & automatic metrics
meeting_2026-06-16_sarcasm_sign.pptx   # Meeting 2: judge architecture & prompt sensitivity
meeting_2026-07-09_sarcasm_sign.pptx   # Meeting 3 (draft)
meeting_2026-07-16_sarcasm_sign.pptx   # Meeting 3 (final): human validation, Alt-Test,
                                        # error analysis, and next steps
```

These files document the development of the methodology, selected models, prompt versions, and results.

### `docs/methodology_from_meetings.md`

Summarizes the methodology discussed in the meetings:

- how the dataset was prepared
- which models were selected
- how outputs were generated
- how evaluation was performed
- how prompt sensitivity was tested

### `docs/pipeline.md`

Explains the project pipeline step by step.

### `docs/results_from_meetings.md`

Summarizes results from the meetings, including model comparisons, prompt sensitivity results, and examples of good/bad interpretations.

---

## `prompts/`

The `prompts` folder stores all prompt templates as `.txt` files.

This keeps prompt engineering separate from the Python code.

### `prompts/generation/`

Prompts used to generate sarcasm interpretations.

#### `generation_prompt_v1.txt`

Initial generation prompt.

#### `generation_prompt_v2.txt`

Alternative generation prompt focused on translating the true meaning of the sarcastic tweet.

#### `generation_prompt_v3.txt`

Generation prompt with additional formatting constraints, such as proper spacing and grammatical output.

#### `generation_prompt_v4.txt`

Few-shot generation prompt: adds 3 worked examples (sarcastic tweet -> correct
direct translation) to the instructions. This is the current default used by
both generation scripts (`--prompt` overrides it).

### `prompts/evaluation/`

Prompts used for evaluation.

#### `llm_judge_prompt.txt`

Prompt used by the LLM judge.

The judge receives:

- the original sarcastic sentence
- the model interpretation

It returns:

```text
1 = Incorrect
2 = Partially correct
3 = Correct
```

#### `binary_judge_prompt.txt`

Older or alternative binary evaluation prompt.

It is used for a simpler `0/1` judgment. If the final project mainly uses the `1/2/3` judge, this prompt is kept for reproducibility and comparison.

#### `nli_premise_template.txt` / `nli_hypothesis_template.txt`

Used by `evaluate_with_nli.py`. Each is a plain pass-through of a single
placeholder (`{sarcastic_sentence}` and `{model_interpretation}`
respectively) -- the NLI model just needs the raw sentence and
interpretation as premise/hypothesis.

---

## `src/`

The `src` folder contains all Python code.

```text
src/
├── common/
├── evaluation/
├── generation/
├── postprocessing/
└── preprocessing/
```

---

## `src/common/`

Shared helper functions used by multiple scripts.

### `src/common/file_utils.py`

Shared CSV and filesystem utilities.

Main functions:

- `ensure_parent_dir(path)` — creates a parent folder before saving a file.
- `read_csv_flexible(path, expected_columns)` — reads a CSV file and can normalize files without headers.
- `save_csv(df, path)` — saves a DataFrame as UTF-8-SIG CSV, which works well with Excel.
- `load_all_model_outputs(model_outputs_dir)` — loads every classified CSV under `data/model_outputs/` into one DataFrame with `prompt`/`model` columns.

### `src/common/gemini_client.py`

Creates a configured Gemini model.

Used by:

```text
src/generation/generate_with_gemini.py
```

It loads the Gemini API key from `.env`.

### `src/common/openrouter_client.py`

Creates an OpenRouter client using the OpenAI-compatible API.

Used by:

```text
src/generation/generate_with_openrouter.py
src/evaluation/evaluate_with_llm.py
src/evaluation/check_openrouter_limit.py
```

It loads the OpenRouter API key from `.env`.

### `src/common/json_utils.py`

Helper for extracting JSON arrays from LLM responses.

Main function:

```python
extract_json_array(text)
```

Useful because LLMs sometimes wrap JSON in Markdown code blocks.

### `src/common/prompt_loader.py`

Loads prompts from the `prompts/` folder.

Example:

```python
from src.common.prompt_loader import load_prompt

prompt = load_prompt("generation/generation_prompt_v1.txt")
```

This keeps prompts outside the Python code.

### `src/common/alt_test.py`

The `alt_test()` function and its scoring helpers (`accuracy`, `neg_rmse`,
`sim`) -- the reference implementation from Calderon, Reichart & Dror
(2025). See `docs/alt_test_reference.md` for the citation and how this
project uses it.

### `src/common/__init__.py`

Marks `common/` as a Python package.

---

## `src/preprocessing/`

Code for preparing the dataset before sending it to models.

### `src/preprocessing/clean_dataset.py`

Cleans the original dataset.

Typical role:

- reads the raw test dataset
- keeps only the sarcastic tweet text
- removes unnecessary columns
- removes duplicates if needed
- saves a clean CSV for generation

Run:

```bash
python -m src.preprocessing.clean_dataset \
  --input data/raw/original_test_dataset.csv \
  --output data/processed/clean_sarcastic_sentences.csv
```

If the project does not currently include `data/processed/`, create it or update the output path.

---

## `src/generation/`

Code that sends sarcastic tweets to language models and saves their interpretations.

### `src/generation/generate_with_gemini.py`

Generates interpretations using Gemini.

It:

- reads a cleaned CSV
- loads a generation prompt (`--prompt`, defaults to the Prompt 4 few-shot
  version: `generation/generation_prompt_v4.txt`)
- sends each tweet to Gemini
- saves the output after each row
- supports row ranges for partial runs

Run:

```bash
python -m src.generation.generate_with_gemini \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_new/gemini.csv \
  --prompt generation/generation_prompt_v2.txt \
  --start-row 0 \
  --end-row 265 \
  --sleep 2
```

### `src/generation/generate_with_openrouter.py`

Generates interpretations using an OpenRouter model.

It can be used for Nvidia, Liquid, GPT-OSS or any OpenRouter-supported model.
Same `--prompt` option as the Gemini script.

Run for Nvidia:

```bash
python -m src.generation.generate_with_openrouter \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_new/nvidia.csv \
  --model nvidia/nemotron-nano-9b-v2:free \
  --prompt generation/generation_prompt_v4.txt \
  --start-row 0 \
  --end-row 265 \
  --sleep 1
```

Run for Liquid:

```bash
python -m src.generation.generate_with_openrouter \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_new/liquid.csv \
  --model liquid/lfm-2.5-1.2b-thinking:free \
  --prompt generation/generation_prompt_v4.txt \
  --start-row 0 \
  --end-row 265 \
  --sleep 1
```

---

## `src/evaluation/`

Code for evaluating model interpretations.

### `src/evaluation/evaluate_with_llm.py`

Main evaluation script.

It uses an LLM judge to assign a score to each model interpretation.

Scores:

```text
1 = Incorrect
2 = Partially correct
3 = Correct
```

It supports:

- evaluating a single CSV
- evaluating all CSV files under `data/model_outputs/`
- batch evaluation
- retrying when the API fails
- saving progress continuously

Evaluate one file:

```bash
python -m src.evaluation.evaluate_with_llm \
  --input data/model_outputs/experiment_04/nvidia_run_04.csv \
  --model openai/gpt-oss-20b:free \
  --batch-size 10
```

Evaluate all model output files:

```bash
python -m src.evaluation.evaluate_with_llm \
  --directory data/model_outputs \
  --model openai/gpt-oss-20b:free \
  --batch-size 10
```

### `src/evaluation/evaluate_with_nli.py`

Alternative automatic evaluation using an NLI model.

It checks whether the generated interpretation is entailed by the sarcastic sentence representation.

This is not necessarily the main evaluation method, but it is useful as an experimental comparison.
Defaults to the model in `config.models.JUDGE_MODELS["nli"]`, overridable via `--model`.

Run:

```bash
python -m src.evaluation.evaluate_with_nli \
  --input data/model_outputs/experiment_04/nvidia_run_04.csv \
  --output data/model_outputs/experiment_04/nvidia_nli.csv
```

---

## `src/tools/`

Utility scripts that support the pipeline but aren't part of the research
pipeline itself.

### `src/tools/check_openrouter_limit.py`

Checks OpenRouter API key usage/limit -- useful before a long generation or
evaluation run to confirm there's enough quota left.

Run:

```bash
python -m src.tools.check_openrouter_limit
```

---

## `src/postprocessing/`

Code that runs after generation and evaluation.

This folder creates summaries, metrics and manual scoring files.

### `src/postprocessing/summarize_classifications.py`

Summarizes `classification` scores across model output CSV files.

It calculates:

- average score
- median score
- count of score 1
- count of score 2
- count of score 3
- total number of classified rows

Run:

```bash
python -m src.postprocessing.summarize_classifications \
  --outputs-dir data/model_outputs \
  --output data/summaries/classification_summary.csv
```

### `src/postprocessing/calculate_text_metrics.py`

Calculates automatic text-overlap and novelty metrics.

Metrics:

- BLEU
- ROUGE-1
- ROUGE-2
- PINC
- Combined: `PINC * sigmoid(BLEU)`

Run:

```bash
python -m src.postprocessing.calculate_text_metrics \
  --input data/model_outputs/experiment_04/nvidia.csv \
  --output data/summaries/text_metrics_nvidia_run_01.csv
```

### `src/postprocessing/create_manual_sample.py`

Creates a random sample for manual scoring.

It samples examples across model output files and creates columns for human scores.

Run:

```bash
python -m src.postprocessing.create_manual_sample \
  --outputs-dir data/model_outputs \
  --output data/manual_scoring/random_70_for_manual_scoring.csv \
  --sample-size 70 \
  --seed 42
```

### `src/postprocessing/run_alt_test.py`

Runs the alt-test on `data/alt_test/humans_annotations.json` and
`llm_annotations.json`, printing the Winning Rate and Advantage Probability
(see `docs/alt_test_reference.md`).

Run:

```bash
python -m src.postprocessing.run_alt_test
```

### Deeper statistical analysis

A further set of scripts (added after the 2026-07-16 meeting) analyzes the
full classified dataset in more depth -- see `docs/project_structure.md` for
full details on each one. All figures save to `data/summaries/figures/`.

```bash
python -m src.postprocessing.summarize_text_metrics   # BLEU/ROUGE/PINC, every prompt/model
python -m src.postprocessing.plot_text_metrics        # boxplots of the above
python -m src.postprocessing.significance_tests       # Kruskal-Wallis: does prompt/model matter?
python -m src.postprocessing.correlation_heatmap      # Spearman: structural metrics vs. quality
python -m src.postprocessing.linguistic_analysis      # sentence length & word-overlap plots
python -m src.postprocessing.human_llm_agreement      # Fleiss' Kappa, human vs. ChatGPT comparison
python -m src.postprocessing.extract_case_studies     # agreement/discrepancy case studies
```

---

## Installation

### Step 1: Open the project

Open the project folder in VS Code or PyCharm.

Make sure the terminal is opened at the root folder:

```bash
cd sarcasm
```

### Step 2: Create a virtual environment

Mac / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create `.env`

Create a file named `.env` in the root folder.

```env
OPENROUTER_API_KEY=your_openrouter_key_here
GEMINI_API_KEY=your_gemini_key_here
```

Do not upload `.env` to GitHub.

---

## Recommended Run Order

### 1. Clean the dataset

```bash
python -m src.preprocessing.clean_dataset \
  --input data/raw/original_test_dataset.csv \
  --output data/processed/clean_sarcastic_sentences.csv
```

### 2. Generate Gemini outputs

`experiment_01`-`experiment_04` already hold real results from past runs --
use a new experiment folder (e.g. `experiment_new`) for a fresh run so you
don't overwrite them.

```bash
python -m src.generation.generate_with_gemini \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_new/gemini.csv \
  --start-row 0 \
  --end-row 265
```

### 3. Generate OpenRouter outputs

Nvidia:

```bash
python -m src.generation.generate_with_openrouter \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_new/nvidia.csv \
  --model nvidia/nemotron-nano-9b-v2:free \
  --start-row 0 \
  --end-row 265
```

Liquid:

```bash
python -m src.generation.generate_with_openrouter \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_new/liquid.csv \
  --model liquid/lfm-2.5-1.2b-thinking:free \
  --start-row 0 \
  --end-row 265
```

### 4. Evaluate outputs with LLM judge

```bash
python -m src.evaluation.evaluate_with_llm \
  --directory data/model_outputs \
  --model openai/gpt-oss-20b:free \
  --batch-size 10
```

### 5. Summarize classifications

```bash
python -m src.postprocessing.summarize_classifications \
  --outputs-dir data/model_outputs \
  --output data/summaries/classification_summary.csv
```

### 6. Calculate automatic metrics

```bash
python -m src.postprocessing.calculate_text_metrics \
  --input data/model_outputs/experiment_04/nvidia_run_04.csv \
  --output data/summaries/text_metrics_nvidia_run_04.csv
```

### 7. Create manual scoring sample

```bash
python -m src.postprocessing.create_manual_sample \
  --outputs-dir data/model_outputs \
  --output data/manual_scoring/random_70_for_manual_scoring.csv \
  --sample-size 70 \
  --seed 42
```

### 8. Check whether the LLM judge can replace human annotators (Alt-Test)

Requires human annotations already collected in
`data/alt_test/humans_annotations.json` (see `docs/alt_test_reference.md`).

```bash
python -m src.postprocessing.run_alt_test
```

---

## CSV Formats

### Model output CSV

Before evaluation:

```text
sarcastic_sentence,model_interpretation
```

After evaluation:

```text
sarcastic_sentence,model_interpretation,classification
```

### Classification summary

```text
experiment,model,average,median,count_1,count_2,count_3,total
```

### Manual scoring sample

```text
experiment
sarcastic_sentence
gemini_interpretation
gemini_score
nvidia_interpretation
nvidia_score
liquid_interpretation
liquid_score
```

---

## Prompt Experiments

The project includes four generation prompt versions.

They are stored in:

```text
prompts/generation/
```

The folders under `data/model_outputs/` represent experiment runs, usually corresponding to different prompt versions:

```text
experiment_01
experiment_02
experiment_03
experiment_04
```

When documenting results, make sure to record which prompt was used for each experiment.

---

## Important Notes

- Never upload `.env` to GitHub.
- Do not manually edit files inside `data/raw/`.
- Save outputs by experiment so older results are not overwritten.
- Run commands from the project root using `python -m ...`.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run commands from the project root using `python -m`.

Correct:

```bash
python -m src.generation.generate_with_gemini
```

Not recommended:

```bash
python src/generation/generate_with_gemini.py
```

### Missing API key

If you see:

```text
OPENROUTER_API_KEY is missing
```

or:

```text
GEMINI_API_KEY is missing
```

check that `.env` exists in the root folder and contains the correct key.

### Check OpenRouter limit

```bash
python -m src.tools.check_openrouter_limit
```

### Prompt file not found

Check that the prompt path passed to `load_prompt(...)` (or `--prompt`)
matches the folder structure, e.g. `generation/generation_prompt_v4.txt`,
not just `generation_prompt_v4.txt`.

---

## Suggested Future Improvements

- Add `docs/reproduction.md` with exact commands for reproducing final results.
- Add logging instead of using `print()`.
- Save per-sentence BLEU/ROUGE/PINC metrics in addition to summary metrics.
- Add tests for `common/` helper functions.

---

## Quick Start

After installing dependencies and adding `.env`, the shortest full flow is:

```bash
python -m src.preprocessing.clean_dataset \
  --input data/raw/original_test_dataset.csv \
  --output data/processed/clean_sarcastic_sentences.csv

python -m src.generation.generate_with_gemini \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_new/gemini.csv

python -m src.evaluation.evaluate_with_llm \
  --input data/model_outputs/experiment_new/gemini.csv

python -m src.postprocessing.summarize_classifications \
  --outputs-dir data/model_outputs \
  --output data/summaries/classification_summary.csv
```

This runs one complete path: clean data → generate Gemini interpretations → evaluate → summarize.
