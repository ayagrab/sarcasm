# Sarcasm Interpretation & Detection

A two-phase NLP project on sarcastic English text, complete end to end.
Full results, methodology, and conclusions for both phases:
**[`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md)**.

- **Phase 1 — Interpretation** (`src/generation/`, `src/evaluation/`,
  `src/postprocessing/`): given a sarcastic tweet, have an LLM rewrite it
  as a clear, sincere statement, and evaluate how well that rewrite
  captures the intended meaning — automatically, by an LLM judge, and by
  human annotators (including whether the LLM judge can validly replace
  human annotators, via the Alt-Test).
- **Phase 2 — Detection** (`src/classification/`): given a short English
  text, predict whether it's sarcastic at all — a 6-way comparison of
  classical ML, LLM zero/few-shot/reasoning prompting, DSPy-optimized
  prompting, and a fine-tuned Transformer encoder. Started after Phase 1
  repeatedly found that models often can't tell a sentence is sarcastic
  in the first place, which no amount of rewriting-prompt refinement can
  fix — see `docs/project_history.md` for the full narrative of that
  pivot.

Both phases are complete: every result below is real, executed, and
recorded — nothing is a plan or a placeholder.

```text
Phase 1 — Interpretation
Original sarcastic tweet -> LLM generates a sincere interpretation
        -> evaluated (automatic metrics + LLM judge + humans)
        -> summarized, statistically analyzed, compared across models/prompts

Phase 2 — Detection
Short English text -> 6 competing approaches (M1-M6) each predict
sarcastic / not_sarcastic -> compared on a shared, sealed test split
```

---

## Repository structure

```text
sarcasm/
|
├── config/          # project-wide settings and model-ID constants
├── configs/         # one JSON config per Stage B experiment/method
├── data/            # datasets and result files (see docs/project_structure.md)
├── docs/            # documentation (see "Documentation" section below)
├── logs/            # raw stdout logs from Stage B experiment runs
├── models/          # trained checkpoints (gitignored -- not in version control)
├── prompts/         # every prompt template, as plain .txt files
├── results/         # per-experiment metrics/predictions (Stage B)
├── scripts/         # GPU-VM workflow: sync, verification, experiment chains
├── src/             # all Python code, one subfolder per pipeline stage (see src/README.md)
├── tests/           # pytest suite (no real API calls, no model downloads)
├── conftest.py      # makes `config`/`src` importable from tests/
├── .env.example     # template for your local .env (copy, then fill in)
├── .gitignore
├── README.md               # this file
├── PROJECT_SUMMARY.md      # full project results, methodology, and conclusions (both phases)
├── EXPERIMENT_LOG.md       # detailed experiment-by-experiment audit trail
├── requirements.txt              # runtime dependencies
├── requirements-classification.txt  # + Stage B (classification) dependencies
└── requirements-dev.txt          # + testing dependencies
```

For a detailed, file-by-file explanation of every folder, see
**[`docs/project_structure.md`](docs/project_structure.md)**.

## Documentation

| Document | What it covers |
|---|---|
| [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) | Full project results, methodology, and conclusions for *both* phases (interpretation and detection) -- the main deliverable |
| [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | Detailed technical record for Stage B, organized by method (M1-M6) |
| [`docs/pipeline.md`](docs/pipeline.md) | Technical, stage-by-stage map of the interpretation pipeline and which stages need an API key or model download |
| [`docs/methodology.md`](docs/methodology.md) | *How* the interpretation pipeline's dataset, models, prompts, and evaluation methods were chosen |
| [`docs/results.md`](docs/results.md) | *What was found* in the interpretation pipeline: metrics, Alt-Test outcome, significance tests, case studies |
| [`docs/project_history.md`](docs/project_history.md) | Chronological, meeting-by-meeting narrative of the project's decisions, including the pivot to sarcasm detection |
| [`docs/alt_test_reference.md`](docs/alt_test_reference.md) | What the Alt-Test is, its source paper, and how it's used here |
| [`docs/finetuning_plan.md`](docs/finetuning_plan.md) | The original sarcasm-detection fine-tuning plan -- superseded by the fuller Stage B comparison, kept for planning history |
| [`docs/project_structure.md`](docs/project_structure.md) | Every folder and file in the repository, explained |
| [`src/README.md`](src/README.md) | Same level of detail as `project_structure.md`, but scoped to the code in `src/` only |
| [`docs/validation.md`](docs/validation.md) | What has been executed, mocked, or still needs real credentials/models (interpretation pipeline) |

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
# Base (Phase 1, always needed):
pip install -r requirements.txt
# + Phase 2 (detection) extras, needed only for src/classification/:
pip install -r requirements-classification.txt
# + testing dependencies, if you want to run the test suite:
pip install -r requirements-dev.txt
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

## Running Phase 1 — the interpretation pipeline

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

## Running Phase 2 — the detection pipeline

Needs `requirements-classification.txt` installed (`dspy`, `accelerate`,
`sentencepiece` — see step 4 of Installation above) and, for the LLM/DSPy
approaches (M2–M5) and the Transformer fine-tune (M6), a CUDA GPU; M1
(TF-IDF) needs neither.

```bash
# 1. Build the canonical dataset from the raw Sarcasm Corpus V2 files
python -m src.classification.data.build_canonical_dataset
python -m src.classification.data.audit_dataset
python -m src.classification.data.make_splits

# 2. Run any approach via its config file under configs/
python -m src.classification.run_experiment --config configs/tfidf.json
python -m src.classification.run_experiment --config configs/llm_zero_shot_qwen_local.json
python -m src.classification.run_experiment --config configs/dspy_mipro_v2.json
python -m src.classification.run_experiment --config configs/transformer_deberta_v3_base.json
```

Every experiment's configuration, metrics, and per-example predictions
land under `results/<experiment_id>/`. See `PROJECT_SUMMARY.md` §13 for
the exact commands and configs used to produce every result in this
repository (including the frozen-configuration TEST evaluations), and
`EXPERIMENT_LOG.md`'s per-method sections (M1–M6) for the full technical
detail behind each one.

---

## Testing and validation

### Run the test suite

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

The suite (in `tests/`) covers both phases, never calls a real API, never
downloads a model, and never needs a GPU -- everything uses temporary
fixtures or mocked responses. **Phase 1:** environment-variable
validation, prompt loading (read-only), CLI `--help` for every script,
text-metric functions, JSON parsing, the Alt-Test algorithm, NLI
label-mapping logic, and mocked Gemini/OpenRouter/LLM-judge/quota-check
request-and-response handling (including error paths). **Phase 2:**
canonical dataset construction, splitting (including the no-leakage
assertion), few-shot demo selection, the shared metrics implementation,
cross-model error analysis, and the LLM/local-HF client with a mocked
model.

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

**Phase 1 (interpretation):**

- **`google-generativeai` (used for Gemini) is deprecated** by Google in
  favor of `google.genai`. It still works today (pinned versions in
  `requirements.txt` keep it functional), but will not receive further
  fixes.
- **The NLI evaluation path (`evaluate_with_nli.py`) has not been executed
  in this environment** -- only statically reviewed and tested via mocked
  label-mapping logic (`tests/test_nli_utils.py`), since running it for
  real requires downloading a model. See `docs/validation.md`.
- **API-backed scripts (Gemini, OpenRouter, LLM judge, quota check)**
  were validated via imports, static review, and mocked responses (see
  `docs/validation.md`); run the real API smoke test above with your own
  keys to confirm live behavior.
- **`prompts/evaluation/nli_premise_template.txt` and
  `nli_hypothesis_template.txt`** are minimal pass-through templates
  (`{sarcastic_sentence}` / `{model_interpretation}` respectively); their
  exact original methodological intent (whether more elaborate wording
  was planned) was not independently confirmed -- review before relying
  on wording changes here.

**Phase 2 (detection):** see `PROJECT_SUMMARY.md` §10 -- the base model
and hardware the LLM results are specific to, the single-seed fine-tuning
run, and the source corpus's own limitations (dated forum text, no
author/conversation metadata for leakage control beyond text-level
deduplication).

## Future work

Both phases are complete -- Phase 1's Alt-Test/human-agreement analysis
and Phase 2's six sealed, one-shot TEST results are both final (see
`PROJECT_SUMMARY.md`). What remains open, for either phase:

- Multi-seed variance estimate for the winning fine-tuned model (M6).
- Manual qualitative characterization of the examples every detection
  method gets wrong (see `PROJECT_SUMMARY.md`'s Future Work for the full
  list, both phases).
- Migrating off the deprecated `google-generativeai` package.
- Additional test coverage (e.g. more end-to-end fixture pipelines).
