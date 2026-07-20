# Sarcasm Interpretation Benchmark

This project evaluates how well different large language models interpret
sarcastic tweets as clear, sincere, non-sarcastic statements, and whether an
LLM judge can reliably replace human annotators for scoring that task.

```text
Original sarcastic tweet
        |
        v
LLM generates a sincere interpretation
        |
        v
The interpretation is evaluated (automatic metrics + LLM judge + humans)
        |
        v
Results are summarized, statistically analyzed, and compared across models/prompts
```

## What's implemented right now

- Cleaning the original sarcasm dataset.
- Generating interpretations with Gemini and with OpenRouter models
  (Nvidia, Liquid, or any other OpenRouter-hosted model).
- Evaluating outputs with an LLM judge (primary) or an NLI model (alternative).
- Automatic text metrics: BLEU, ROUGE, PINC, and a combined score.
- Human annotation workflow and comparison against the LLM judge, including
  the Alt-Test (a statistical test for whether an LLM can replace human
  annotators), Fleiss' Kappa, Kruskal-Wallis significance tests, Spearman
  correlation, and linguistic/structural analysis.

## What's not implemented yet

- **BERT-based sarcasm *detection* fine-tuning** -- planned as the next
  project phase (see `docs/finetuning_plan.md`), but no training code or
  infrastructure exists in this repository yet.
- No virtual-machine or GPU deployment scripts.
- See `docs/project_history.md` for the full narrative of how the project
  arrived at this plan.

---

## Repository structure

```text
sarcasm/
|
├── config/          # project-wide settings and model-ID constants
├── data/            # datasets and result files (see docs/project_structure.md)
├── docs/            # documentation (see "Documentation" section below)
├── prompts/         # every prompt template, as plain .txt files
├── src/             # all Python code, one subfolder per pipeline stage (see src/README.md)
├── tests/           # pytest suite (no real API calls, no model downloads)
├── conftest.py      # makes `config`/`src` importable from tests/
├── .env.example     # template for your local .env (copy, then fill in)
├── .gitignore
├── README.md        # this file
├── requirements.txt        # runtime dependencies
└── requirements-dev.txt    # + testing dependencies
```

For a detailed, file-by-file explanation of every folder, see
**[`docs/project_structure.md`](docs/project_structure.md)**.

## Documentation

| Document | What it covers |
|---|---|
| [`docs/pipeline.md`](docs/pipeline.md) | Technical, stage-by-stage map of the codebase and which stages need an API key or model download |
| [`docs/methodology.md`](docs/methodology.md) | *How* the dataset, models, prompts, and evaluation methods were chosen |
| [`docs/results.md`](docs/results.md) | *What was found*: metrics, Alt-Test outcome, significance tests, case studies |
| [`docs/project_history.md`](docs/project_history.md) | Chronological, meeting-by-meeting narrative of the project's decisions |
| [`docs/alt_test_reference.md`](docs/alt_test_reference.md) | What the Alt-Test is, its source paper, and how it's used here |
| [`docs/finetuning_plan.md`](docs/finetuning_plan.md) | The (not-yet-implemented) next phase: sarcasm-detection fine-tuning |
| [`docs/project_structure.md`](docs/project_structure.md) | Every folder and file, explained |
| [`src/README.md`](src/README.md) | Same level of detail as `project_structure.md`, but scoped to the code in `src/` only |
| [`docs/validation.md`](docs/validation.md) | What has been executed, mocked, or still needs real credentials/models |

A note on presentations: the project's four supervisor-meeting slide decks
(`.pptx` files) have been removed from the repository. Every piece of
unique information they contained -- experiment descriptions, results,
decisions, open questions -- was first extracted into `docs/project_history.md`,
`docs/methodology.md`, and `docs/results.md`. Nothing was lost.

---

## Installation

**Supported Python version:** 3.10+ (developed and tested on 3.10.6).

### 1. Clone the repository

```bash
git clone https://github.com/ayagrab/sarcasm.git
cd sarcasm
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate it

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Windows (cmd.exe):

```cmd
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
# add -r requirements-dev.txt as well if you want to run the test suite:
pip install -r requirements.txt -r requirements-dev.txt
```

### 5. IDE setup (optional but recommended)

In VS Code (or a similar IDE), select `.venv`'s Python interpreter (Command
Palette -> "Python: Select Interpreter" -> choose the one under
`sarcasm/.venv/`), so the editor resolves imports and stops flagging
installed packages as missing. Always run scripts from the repository root
(`sarcasm/`), using `python -m src.<subfolder>.<script>` -- not
`python src/.../script.py` -- so the `config`/`src` imports resolve
correctly. `.venv/` is already git-ignored; do not commit it.

### 6. Set up your API keys

```bash
cp .env.example .env
```

Then edit `.env` and fill in real values:

| Variable | Required by | Get a key at |
|---|---|---|
| `OPENROUTER_API_KEY` | `generate_with_openrouter.py`, `evaluate_with_llm.py` (the LLM judge), `check_openrouter_limit.py` | https://openrouter.ai/keys |
| `GEMINI_API_KEY` | `generate_with_gemini.py` | https://aistudio.google.com/app/apikey |

Never commit `.env` -- it's already in `.gitignore`. Only `.env.example`
(placeholders only) is committed.

If a key is missing or empty, the affected script fails immediately with a
clear message like `RuntimeError: OPENROUTER_API_KEY is missing. Add it to
your .env file.` -- not a confusing traceback.

---

## Running the pipeline

### Which stages need what

| Stage | Needs an API key? | Needs a model download? |
|---|---|---|
| Clean dataset | No | No |
| Generate with Gemini | Yes -- `GEMINI_API_KEY` | No |
| Generate with OpenRouter | Yes -- `OPENROUTER_API_KEY` | No |
| Evaluate with LLM judge | Yes -- `OPENROUTER_API_KEY` | No |
| Evaluate with NLI | No | Yes (downloads a Hugging Face model on first run) |
| Check OpenRouter quota | Yes -- `OPENROUTER_API_KEY` | No |
| Everything under `src.postprocessing.*` (summaries, metrics, Alt-Test, significance tests, plots) | No | No |

Everything in the last row runs on data that's already in the repository --
no key or download needed. See `docs/validation.md` for exactly which
commands have been executed in this environment and which still need real
credentials or the NLI model download to confirm.

### Recommended run order

`experiment_01`-`experiment_04` already hold real results from past runs --
use a new experiment folder (e.g. `experiment_new`) for a fresh generation
run so you don't overwrite them.

```bash
# 1. Clean the dataset (input: data/raw/original_test_dataset.csv)
python -m src.preprocessing.clean_dataset

# 2. Generate interpretations (needs GEMINI_API_KEY / OPENROUTER_API_KEY)
python -m src.generation.generate_with_gemini \
  --output data/model_outputs/experiment_new/gemini.csv

python -m src.generation.generate_with_openrouter \
  --model nvidia/nemotron-nano-9b-v2:free \
  --output data/model_outputs/experiment_new/nvidia.csv

python -m src.generation.generate_with_openrouter \
  --model liquid/lfm-2.5-1.2b-thinking:free \
  --output data/model_outputs/experiment_new/liquid.csv

# 3. Evaluate with the LLM judge (needs OPENROUTER_API_KEY)
python -m src.evaluation.evaluate_with_llm --directory data/model_outputs

# 3b. Or evaluate with NLI instead (needs a one-time model download, no API key)
python -m src.evaluation.evaluate_with_nli \
  --input data/model_outputs/experiment_04/nvidia_run_04.csv \
  --output data/model_outputs/experiment_04/nvidia_nli.csv

# 4. Summarize and analyze (no API key or model download needed for any of these)
python -m src.postprocessing.summarize_classifications
python -m src.postprocessing.summarize_text_metrics
python -m src.postprocessing.plot_text_metrics
python -m src.postprocessing.significance_tests
python -m src.postprocessing.correlation_heatmap
python -m src.postprocessing.linguistic_analysis
python -m src.postprocessing.create_manual_sample
python -m src.postprocessing.run_alt_test
python -m src.postprocessing.human_llm_agreement
python -m src.postprocessing.extract_case_studies
```

Full detail on each script (arguments, what it reads/writes) is in
`docs/project_structure.md` and `docs/pipeline.md`.

### CSV formats

Model output CSV, before evaluation:

```text
sarcastic_sentence,model_interpretation
```

After evaluation (LLM judge adds `classification`; NLI adds `nli_success`):

```text
sarcastic_sentence,model_interpretation,classification
```

---

## Testing and validation

### Run the test suite

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

The suite (in `tests/`) never calls a real API, never downloads a model,
and never touches real project data -- everything uses temporary fixtures
or mocked responses. It covers: environment-variable validation, prompt
loading (read-only), CLI `--help` for every script, text-metric functions,
JSON parsing, the Alt-Test algorithm, NLI label-mapping logic, and mocked
Gemini/OpenRouter/LLM-judge/quota-check request-and-response handling
(including error paths).

### Real API smoke test (requires your own keys)

Once `.env` has real keys, confirm each integration actually works end to
end with a tiny, cheap run:

```bash
# OpenRouter: confirms your key/quota works
python -m src.tools.check_openrouter_limit

# Gemini generation: 1 row
python -m src.generation.generate_with_gemini --start-row 0 --end-row 1 \
  --output /tmp/gemini_smoke_test.csv

# OpenRouter generation: 1 row
python -m src.generation.generate_with_openrouter --start-row 0 --end-row 1 \
  --model nvidia/nemotron-nano-9b-v2:free --output /tmp/openrouter_smoke_test.csv

# LLM judge: evaluate that 1-row file
python -m src.evaluation.evaluate_with_llm --input /tmp/openrouter_smoke_test.csv --batch-size 1
```

You know it passed if each command exits without a traceback and the
output CSV has a non-empty `model_interpretation` (and `classification`,
for the judge step).

### Real NLI validation (requires a one-time model download, no API key)

```bash
python -m src.evaluation.evaluate_with_nli \
  --input data/model_outputs/experiment_04/nvidia_run_04.csv \
  --output /tmp/nli_smoke_test.csv
```

This downloads `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` from Hugging Face
on first run. Confirm it passed by checking `/tmp/nli_smoke_test.csv` has an
`nli_success` column of 0s and 1s, and spot-check a few rows by eye: rows
where the interpretation clearly captures the sarcastic meaning should
mostly be 1.

See `docs/validation.md` for exactly what has and hasn't been verified in
this environment, and why (no real keys, no model download performed here).

---

## Known limitations

- **`google-generativeai` (used for Gemini) is deprecated** by Google in
  favor of `google.genai`. It still works today (pinned versions in
  `requirements.txt` keep it functional), but will not receive further
  fixes. Migrating is future work, not part of this cleanup.
- **The NLI evaluation path (`evaluate_with_nli.py`) has not been executed
  in this environment** -- only statically reviewed and tested via mocked
  label-mapping logic (`tests/test_nli_utils.py`), since running it for
  real requires downloading a model. See `docs/validation.md`.
- **API-backed scripts (Gemini, OpenRouter, LLM judge, quota check) have
  not been re-tested with real credentials in this environment** -- they
  were previously tested with real keys, but this cleanup only validated
  them via imports, static review, and mocked responses (see
  `docs/validation.md`). Run the real API smoke test above once you have
  your own keys.
- **`prompts/evaluation/nli_premise_template.txt` and
  `nli_hypothesis_template.txt`** are minimal pass-through templates
  (`{sarcastic_sentence}` / `{model_interpretation}` respectively) created
  because the code referenced them but the files didn't exist. Their exact
  original methodological intent (whether more elaborate wording was
  planned) was not independently confirmed -- review before relying on
  wording changes here.
- **`data/manual_scoring/aya.numbers`** shows as modified relative to an
  earlier version of the repository; this predates the current cleanup,
  was not investigated, and has been left untouched.

## Future work

- BERT-based sarcasm-detection fine-tuning (see `docs/finetuning_plan.md`) -- planning only, no code yet.
- Virtual-machine / GPU deployment for the above.
- Migrating off the deprecated `google-generativeai` package.
- Additional test coverage (e.g. more end-to-end fixture pipelines).
