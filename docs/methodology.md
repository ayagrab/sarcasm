# Methodology

How the dataset was prepared, which models and prompts were used, and how
outputs were evaluated. For what was actually found, see `results.md`. For
the chronological, meeting-by-meeting narrative behind these decisions, see
`project_history.md`.

## 1. Dataset preparation

The project starts from the original test data used by the SIGN paper
(sarcasm interpretation). The reference/result column is removed, leaving
only the sarcastic tweets that are sent to the models.

```bash
python -m src.preprocessing.clean_dataset
```

Output: `data/processed/clean_sarcastic_sentences.csv` (one column,
`sarcastic_sentence`, deduplicated).

## 2. Model selection

Five candidate models were surveyed first (see `project_history.md` for the
full comparison): Gemini 2.5 Flash Lite, Baidu Qianfan OCR FastFree, Liquid
LFM 2.5-1.2B Thinking, Nvidia Nemotron Nano 9B v2, and OpenAI GPT-OSS 20B.
The project settled on three generator models for the main pipeline:

- **Gemini 2.5 Flash Lite** (Google, via `src.generation.generate_with_gemini`)
- **Nvidia Nemotron Nano 9B v2** (via OpenRouter, `src.generation.generate_with_openrouter`)
- **Liquid LFM 2.5-1.2B Thinking** (via OpenRouter, `src.generation.generate_with_openrouter`)

Baidu and GPT-OSS were used only in the initial survey and are not part of
the ongoing generation pipeline.

## 3. Generation prompts

Each model receives the same sarcastic tweet with one of four prompt
versions, stored in `prompts/generation/`:

- **Prompt 1** (`generation_prompt_v1.txt`) -- plain instruction, no examples.
- **Prompt 2** (`generation_prompt_v2.txt`) -- "translate the true meaning,"
  output only the sentence.
- **Prompt 3** (`generation_prompt_v3.txt`) -- adds formatting/grammar
  constraints.
- **Prompt 4** (`generation_prompt_v4.txt`) -- few-shot, with 3 worked
  examples.

All four use the same placeholder, `{sarcastic_sentence}`, so any version
can be selected via `--prompt` on either generation script (default: Prompt
4). Prompt content itself has not been modified as part of any later
cleanup -- see `validation.md` for confirmation.

```bash
python -m src.generation.generate_with_gemini --prompt generation/generation_prompt_v2.txt ...
```

## 4. Evaluation methods

Three independent evaluation methods were used:

### 4.1 Automatic text-overlap metrics

`src.postprocessing.calculate_text_metrics` (single file) and
`summarize_text_metrics` (all files) compute:

- **BLEU** (precision-oriented n-gram overlap with the source)
- **ROUGE-1 / ROUGE-2** (recall-oriented unigram/bigram overlap)
- **PINC** (novelty: fraction of interpretation words not present in the source)
- **Combined score** = `PINC * sigmoid(BLEU)`

### 4.2 LLM judge

`src.evaluation.evaluate_with_llm` sends the sarcastic sentence and the
model's interpretation to an independent judge model (OpenAI GPT-OSS 20B,
via OpenRouter), using `prompts/evaluation/llm_judge_prompt.txt`. The judge
returns a strict score:

- **1 = Incorrect** -- wrong, literal, unrelated, or misses the sarcasm.
- **2 = Partially correct** -- captures the general idea but misses nuance.
- **3 = Correct** -- accurately captures the intended sarcastic meaning.

An older/alternative binary (0/1) judge prompt
(`prompts/evaluation/binary_judge_prompt.txt`) exists for reproducibility
but is not currently called by any script.

### 4.3 NLI evaluation (alternative)

`src.evaluation.evaluate_with_nli` treats the sarcastic sentence as the
*premise* and the model's interpretation as the *hypothesis*, and checks
whether an NLI model predicts entailment more strongly than contradiction.
This is an experimental alternative to the LLM judge, not the primary
evaluation method. See `validation.md` for what has been verified about this
script without downloading the model.

## 5. Human validation

Because automatic metrics and a single LLM judge may not fully capture
meaning preservation, three team members independently scored a random
sample of 70 tweets across all 4 prompts and all 3 models (see
`data/manual_scoring/`), using the same 1-3 scale as the LLM judge. This
allows direct comparison between human judgment and the automated judge.

## 6. Prompt sensitivity

The same 3 models were run under all 4 prompt versions (the `experiment_01`
through `experiment_04` folders), to test whether wording changes the
outputs' quality -- both automatically (BLEU/ROUGE/PINC,
`Kruskal-Wallis`) and via human/LLM-judge scores.

## 7. Alt-Test: is the LLM judge a valid substitute for human annotators?

The Alt-Test (Calderon, Reichart & Dror, 2025 -- see
`alt_test_reference.md`) is a statistical procedure for justifying the use
of an LLM annotator in place of human annotators. It uses a leave-one-out
strategy: each human annotator is excluded in turn, and the LLM judge is
compared against the *remaining* annotators to see who better represents
them.

- **Winning Rate**: fraction of human annotators the LLM out-performs.
  Passes if `>= 0.5`.
- **Advantage Probability (rho)**: estimated probability the LLM is at
  least as good as a random human annotator.
- **epsilon**: a cost-benefit tolerance for the LLM's efficiency advantage.
  This project used **epsilon = 0.2** (appropriate for expert-level human
  annotators); see `alt_test_reference.md` for why a lower value was
  rejected.

Run: `python -m src.postprocessing.run_alt_test`

## 8. Statistical significance and agreement analysis

Beyond the Alt-Test, several further analyses characterize the relationship
between human judgment, the LLM judge, and the underlying text:

- **Fleiss' Kappa** (`src.postprocessing.human_llm_agreement`): inter-rater
  agreement among the 3 human annotators, to establish how subjective the
  task is for humans themselves.
- **Kruskal-Wallis** (`src.postprocessing.significance_tests`): whether
  prompt choice and model choice significantly affect the LLM-judge
  classification score.
- **Spearman correlation** (`src.postprocessing.correlation_heatmap`):
  whether simple structural features (sentence length, word overlap)
  predict quality score.
- **Linguistic analysis** (`src.postprocessing.linguistic_analysis`):
  per-model rewriting strategies (expansion, stability, or aggressive
  rewriting) and how word overlap relates to quality.
- **Case studies** (`src.postprocessing.extract_case_studies`): the
  clearest agreement and disagreement cases between human and LLM-judge
  scores, for qualitative review.

All of the above are computed directly from `data/alt_test/*.json` and
`data/model_outputs/` -- no separate/duplicate copies of this data are
stored.
