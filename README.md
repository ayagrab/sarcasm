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
│   ├── manual_scoring/
│   ├── model_outputs/
│   │   ├── experiment_01/
│   │   ├── experiment_02/
│   │   ├── experiment_03/
│   │   └── experiment_04/
│   ├── raw/
│   └── summaries/
│
├── docs/
│   ├── meeting_slides/
│   ├── methodology_from_meetings.md
│   ├── pipeline.md
│   └── results_from_meetings.md
│
├── prompts/
│   ├── evaluation/
│   │   ├── binary_judge_prompt.txt
│   │   └── llm_judge_prompt.txt
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
│   └── preprocessing/
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

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
- results folder
- API keys loaded from `.env`
- default model names
- shared constants such as CSV encoding and classification column

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

### `data/raw/`

Original input data.

Current important file:

```text
original_test_dataset.csv
```

This should be the dataset before project-specific cleaning. Do not manually edit files in `raw/`.

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

Contains files prepared for manual evaluation.

Example:

```text
random_70_for_manual_scoring.csv
```

This file contains random examples and empty score columns for human annotation.

### `data/summaries/`

Contains summary tables and final metrics.

Examples:

```text
classification_summary.csv
text_metrics_nvidia_run_01.csv
text_metrics_summary.csv
```

Use this folder for aggregated result files.

---

## `docs/`

The `docs` folder explains the research and project decisions.

### `docs/meeting_slides/`

Contains the original meeting presentations.

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

Additional generation prompt version tested as part of the prompt sensitivity experiments.

This is not necessarily the final prompt; it is simply the fourth tested version.

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
- loads a generation prompt
- sends each tweet to Gemini
- saves the output after each row
- supports row ranges for partial runs

Run:

```bash
python -m src.generation.generate_with_gemini \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_04/gemini.csv \
  --start-row 0 \
  --end-row 265 \
  --sleep 2
```

### `src/generation/generate_with_openrouter.py`

Generates interpretations using an OpenRouter model.

It can be used for Nvidia, Liquid, GPT-OSS or any OpenRouter-supported model.

Run for Nvidia:

```bash
python -m src.generation.generate_with_openrouter \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_04/nvidia.csv \
  --model nvidia/nemotron-nano-9b-v2:free \
  --start-row 0 \
  --end-row 265 \
  --sleep 1
```

Run for Liquid:

```bash
python -m src.generation.generate_with_openrouter \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_04/liquid.csv \
  --model liquid/lfm-2.5-1.2b-thinking:free \
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
  --input data/model_outputs/experiment_04/nvidia.csv \
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

Run:

```bash
python -m src.evaluation.evaluate_with_nli \
  --input data/model_outputs/experiment_04/nvidia.csv \
  --output data/model_outputs/experiment_04/nvidia_nli.csv
```

### `src/evaluation/check_openrouter_limit.py`

Utility script that checks OpenRouter API key usage/limit.

Run:

```bash
python -m src.evaluation.check_openrouter_limit
```

Note: this file is a tool, not part of the core research pipeline.

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

```bash
python -m src.generation.generate_with_gemini \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_04/gemini.csv \
  --start-row 0 \
  --end-row 265
```

### 3. Generate OpenRouter outputs

Nvidia:

```bash
python -m src.generation.generate_with_openrouter \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_04/nvidia.csv \
  --model nvidia/nemotron-nano-9b-v2:free \
  --start-row 0 \
  --end-row 265
```

Liquid:

```bash
python -m src.generation.generate_with_openrouter \
  --input data/processed/clean_sarcastic_sentences.csv \
  --output data/model_outputs/experiment_04/liquid.csv \
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
  --input data/model_outputs/experiment_04/nvidia.csv \
  --output data/summaries/text_metrics_nvidia_run_01.csv
```

### 7. Create manual scoring sample

```bash
python -m src.postprocessing.create_manual_sample \
  --outputs-dir data/model_outputs \
  --output data/manual_scoring/random_70_for_manual_scoring.csv \
  --sample-size 70 \
  --seed 42
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
- If the code still tries to load `interpret_sarcasm_prompt.txt`, update it to the new path, for example `generation/generation_prompt_v4.txt`.

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
python -m src.evaluation.check_openrouter_limit
```

### Prompt file not found

Check that the prompt path in the code matches the folder structure.

Example:

```text
prompts/generation/generation_prompt_v4.txt
```

---

## Suggested Future Improvements

- Move `check_openrouter_limit.py` from `evaluation/` to a separate `tools/` folder.
- Add `processed/` under `data/` if the cleaned dataset is regenerated often.
- Add `docs/project_structure.md` for detailed folder documentation.
- Add `docs/reproduction.md` with exact commands for reproducing final results.
- Add logging instead of using `print()`.
- Add a `--prompt` argument to generation scripts so each prompt version can be selected from the command line.
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
  --output data/model_outputs/experiment_04/gemini.csv

python -m src.evaluation.evaluate_with_llm \
  --input data/model_outputs/experiment_04/gemini.csv

python -m src.postprocessing.summarize_classifications \
  --outputs-dir data/model_outputs \
  --output data/summaries/classification_summary.csv
```

This runs one complete path: clean data → generate Gemini interpretations → evaluate → summarize.
