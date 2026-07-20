# Validation Report

What has actually been run, mocked, or only statically reviewed in this
repository cleanup pass -- described honestly, per component. This
complements `README.md`'s quick-start instructions.

## Verification levels used below

1. **Executed successfully** -- run locally with real (or real-shaped) input; completed; output inspected.
2. **Executed with fixtures** -- run with a small local test fixture; output inspected.
3. **Validated with mocks** -- external API replaced with a controlled mock; request/response/error handling tested.
4. **CLI and import validated** -- imports succeed, `--help` runs, required files are found.
5. **Statically reviewed only** -- inspected, but not executed (missing dependency/credential/model/service).
6. **Not validated** -- could not be meaningfully checked.

## Component-by-component results

| Component | Command(s) run | Verification level | Notes / remaining limitation |
|---|---|---|---|
| `src.preprocessing.clean_dataset` | `python -m src.preprocessing.clean_dataset` | Executed successfully | Ran against the real `data/raw/original_test_dataset.csv`; produces 265 unique sentences. |
| `src.generation.generate_with_gemini` | `--help`; `generate_interpretations()` called directly with a mocked model against the real default input file | Executed with fixtures + CLI validated | Never called the real Gemini API in this session (no key available). Request construction and response/error handling verified via `tests/test_generation_mocked.py`. |
| `src.generation.generate_with_openrouter` | same pattern as above | Executed with fixtures + CLI validated | Never called the real OpenRouter API in this session. |
| `src.evaluation.evaluate_with_llm` | `--help`; `classify_batch`/`evaluate_file` exercised with a mocked OpenRouter client (success, markdown-fenced JSON, malformed JSON, wrong score count, out-of-range score, retry-then-succeed) | Validated with mocks | Never called the real judge model in this session. See `tests/test_evaluate_with_llm_mocked.py`. |
| `src.evaluation.evaluate_with_nli` | `--help`; import check; entailment/contradiction label-mapping logic tested in isolation (`tests/test_nli_utils.py`) with mocked `id2label`/probabilities, including case-insensitive labels, reordered labels, generic `LABEL_0`-style placeholders (documented fallback), and unresolvable labels (raises a clear error) | CLI and import validated; label-mapping validated with mocks | The NLI model (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`) was **not downloaded or run** in this environment -- see "Real NLI validation" in `README.md` for how to do that yourself. |
| `src.tools.check_openrouter_limit` | import check; `main()` exercised with a mocked `requests.get` (success and 401 auth failure) | Validated with mocks | Never called the real OpenRouter API in this session. |
| `src.postprocessing.summarize_classifications` | `python -m src.postprocessing.summarize_classifications` | Executed successfully | Ran against real `data/model_outputs/`; output inspected. |
| `src.postprocessing.calculate_text_metrics` | run against a real experiment file; unit-tested on a tiny fixture | Executed successfully | |
| `src.postprocessing.summarize_text_metrics` / `plot_text_metrics` | run against real `data/model_outputs/` | Executed successfully | Reproduces the same BLEU/ROUGE/PINC values (within floating-point noise) as the project's own prior analysis. |
| `src.postprocessing.create_manual_sample` | run against real `data/model_outputs/`; `find_one_csv` unit-tested for 0-match/2-match error paths | Executed successfully | |
| `src.common.alt_test` / `run_alt_test.py` | run against the real `data/alt_test/*.json` | Executed successfully | Reproduces the project's documented result *exactly*: Winning Rate 0.67, Advantage Probability 0.77, "Dropped 3 instances with less than 2 annotators." Also unit-tested on constructed synthetic scenarios (`tests/test_alt_test.py`). |
| `src.postprocessing.significance_tests` | run against real `data/model_outputs/` | Executed successfully | Reproduces the documented Kruskal-Wallis statistics exactly: prompt effect statistic=156.699, p=9.4496e-34; model effect statistic=328.303, p=5.1281e-72. |
| `src.postprocessing.correlation_heatmap` / `linguistic_analysis` | run against real `data/model_outputs/` | Executed successfully | Correlations and rewriting-pattern findings match the documented results. |
| `src.postprocessing.human_llm_agreement` | run against real `data/alt_test/*.json` + `data/model_outputs/` | Executed successfully | Reproduces the documented Fleiss' Kappa (0.282) and agreement rates (Liquid 67.1%, Nvidia 31.4%) exactly. Also unit-tested with synthetic fixtures. |
| `src.postprocessing.extract_case_studies` | run against real data | Executed successfully | Reproduces the documented 104/210 (49.5%) perfect-agreement rate exactly, at the individual-row level. |
| `src.common.file_utils` | run as part of every script above; unit-tested directly | Executed successfully | **A real bug was found and fixed during this cleanup**: `read_csv_flexible` crashed (`ValueError: Length mismatch`) on the real `data/processed/clean_sarcastic_sentences.csv` -- the default input for both generation scripts -- because its header-detection heuristic misfired on a legitimately-headered file with fewer columns than expected. Fixed and re-verified against the real file; see git history for the exact change. |
| `src.common.prompt_loader` | unit-tested against the real prompt files (read-only) | Executed successfully | No prompt content was modified while testing. |
| `src.common.json_utils` | unit-tested | Executed successfully | Pure, deterministic function. |
| `src.common.nli_utils` | unit-tested with mocked probabilities/labels | Executed successfully | See `evaluate_with_nli` row above -- this is the extracted, testable piece of that script's logic. |
| `src.common.gemini_client` / `openrouter_client` | unit-tested for the missing-key error path | Validated with mocks | Never constructed a real client with a real key in this session. |
| Full test suite | `pytest` | Executed successfully | 89 tests, all passing, no real API calls or model downloads. |

## What still requires real credentials

None of the API-backed integrations above were re-tested with real Gemini
or OpenRouter credentials during this cleanup -- no keys were available.
They were previously tested with real credentials (per project history),
but that has not been repeated here. Use the "Real API smoke test" section
in `README.md` once you have your own keys.

## What still requires the NLI model download

`evaluate_with_nli.py`'s end-to-end behavior (tokenization, actual model
inference, and the label-mapping logic applied to real model output) has
not been executed in this environment, since it requires downloading
`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` from Hugging Face. The
label-mapping logic itself *has* been verified in isolation with mocked
model outputs (see the table above). Use the "Real NLI validation" section
in `README.md` to complete this yourself.

## Prompt content

No prompt file or prompt string embedded in code was modified during this
cleanup pass. Verified via `git diff -- prompts/` showing no changes
relative to the last commit before this pass began.

(Historical note: an earlier cleanup pass, before prompt-content
modification was explicitly disallowed, renamed the `{tweet}` placeholder
to `{sarcastic_sentence}` in `generation_prompt_v1.txt`/`v2.txt`/`v3.txt`
for consistency with `v4.txt` and with the rest of the codebase. That
change is already committed and pushed to `main` and was not altered,
reverted, or added to during this pass.)
