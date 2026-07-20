# Pipeline

This is the technical, stage-by-stage map of the codebase. For *why* each
stage exists and how it was designed, see `methodology.md`. For what was
found at each stage, see `results.md`.

The project has two independent pipelines that share the same raw dataset
and infrastructure:

1. **Interpretation pipeline** (implemented): generate a non-sarcastic
   interpretation for each sarcastic tweet, then evaluate it automatically
   and by a human/LLM judge.
2. **Detection fine-tuning** (planned, not implemented): see
   `finetuning_plan.md`.

## Interpretation pipeline

```text
data/raw/original_test_dataset.csv
        |
        v
src.preprocessing.clean_dataset
        |
        v
data/processed/clean_sarcastic_sentences.csv
        |
        v
src.generation.generate_with_gemini / generate_with_openrouter
        |
        v
data/model_outputs/experiment_XX/*.csv
        |
        v
src.evaluation.evaluate_with_llm   (main)
src.evaluation.evaluate_with_nli   (optional, alternative)
        |
        v
model output CSVs with a `classification` column
        |
        v
src.postprocessing.*  (summaries, metrics, human validation, Alt-Test)
        |
        v
data/summaries/  and  data/manual_scoring/
```

## Code mapping

| Pipeline step | Code |
|---|---|
| Clean dataset | `src.preprocessing.clean_dataset` |
| Gemini generation | `src.generation.generate_with_gemini` |
| OpenRouter generation | `src.generation.generate_with_openrouter` |
| LLM judge evaluation | `src.evaluation.evaluate_with_llm` |
| NLI evaluation (alternative) | `src.evaluation.evaluate_with_nli` |
| OpenRouter quota check (tool) | `src.tools.check_openrouter_limit` |
| Classification summary | `src.postprocessing.summarize_classifications` |
| Automatic text metrics (single file) | `src.postprocessing.calculate_text_metrics` |
| Automatic text metrics (all files) | `src.postprocessing.summarize_text_metrics` + `plot_text_metrics` |
| Manual scoring sample | `src.postprocessing.create_manual_sample` |
| Alt-Test (LLM-judge vs. human validity) | `src.postprocessing.run_alt_test` (algorithm: `src.common.alt_test`) |
| Prompt/model significance (Kruskal-Wallis) | `src.postprocessing.significance_tests` |
| Structural-metric correlation (Spearman) | `src.postprocessing.correlation_heatmap` |
| Sentence-length / word-overlap analysis | `src.postprocessing.linguistic_analysis` |
| Human vs. LLM-judge agreement (Fleiss' Kappa, confusion matrix) | `src.postprocessing.human_llm_agreement` |
| Agreement/discrepancy case studies | `src.postprocessing.extract_case_studies` |

## Experiment folders

Repeated prompt/model runs are stored as:

```text
data/model_outputs/experiment_01/
data/model_outputs/experiment_02/
data/model_outputs/experiment_03/
data/model_outputs/experiment_04/
```

Each experiment folder corresponds to one generation prompt version
(`prompts/generation/generation_prompt_v0N.txt`) and contains one CSV per
generator model (Gemini, Nvidia, Liquid).

## Which stages need what

| Stage | Needs a real API key? | Needs a model download? |
|---|---|---|
| Clean dataset | No | No |
| Generate with Gemini | Yes (`GEMINI_API_KEY`) | No |
| Generate with OpenRouter | Yes (`OPENROUTER_API_KEY`) | No |
| Evaluate with LLM judge | Yes (`OPENROUTER_API_KEY`) | No |
| Evaluate with NLI | No | Yes (downloads a Hugging Face model on first run) |
| Check OpenRouter quota | Yes (`OPENROUTER_API_KEY`) | No |
| Everything under `src.postprocessing.*` | No | No |

See `README.md` for exact commands, and `validation.md` for what has and
has not been executed in this environment.
