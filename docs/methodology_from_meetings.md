# Methodology Derived from Supervisor Meetings

This document summarizes the experimental pipeline documented in the meeting slides and maps it to the refactored codebase.

## Stage 1: Dataset preparation

The project uses the original test data from the sarcasm interpretation paper. The original result/reference column is removed, leaving only the sarcastic tweets that are sent to the models.

Relevant code:

```bash
python -m src.preprocessing.clean_dataset
```

Expected output:

```text
data/processed/clean_sarcastic_sentences.csv
```

## Stage 2: Model generation

Each model receives the same sarcastic tweets and is asked to produce a natural, non-sarcastic interpretation.

The base generation prompt used in the early phase was:

```text
Task: Interpret the following sarcastic tweet.
Goal: Write one natural, non-sarcastic sentence that preserves the intended meaning.
Rules: Output only the interpretation. Do not explain. Do not mention sarcasm.
Tweet:
```

In later prompt-sensitivity tests, several prompt versions were compared. These versions are stored in `prompts/`.

Relevant code:

```bash
python -m src.generation.generate_with_gemini
python -m src.generation.generate_with_openrouter
```

## Stage 3: Automatic text metrics

The project calculates metrics inspired by the original paper:

| Metric | Purpose |
|---|---|
| BLEU | Precision-oriented overlap with the reference. |
| ROUGE-1 | Recall-oriented unigram overlap. |
| ROUGE-2 | Recall-oriented bigram overlap. |
| PINC | Novelty/paraphrasing relative to the source/reference. |
| PINC × sigmoid(BLEU) | Combined score balancing novelty and alignment. |

Relevant code:

```bash
python -m src.analysis.calculate_text_metrics --input <model_output.csv>
```

## Stage 4: LLM judge evaluation

Because automatic metrics may not fully capture meaning preservation, the later pipeline adds an independent LLM judge. The judge compares the original sarcastic sentence with the model interpretation and assigns a score.

Scoring scale:

| Score | Meaning |
|---|---|
| 1 | Incorrect: the interpretation is wrong, literal, unrelated, or misses the sarcastic meaning. |
| 2 | Partially correct: captures the general idea but misses context, tone, nuance, or important details. |
| 3 | Correct: accurately captures the intended sarcastic meaning. |

Relevant code:

```bash
python -m src.evaluation.evaluate_with_llm
```

## Stage 5: Prompt sensitivity

The project compares several prompt variants to test whether model performance changes when the wording of the instruction changes.

The current data structure stores this as experiment folders:

```text
data/model_outputs/experiment_01/
data/model_outputs/experiment_02/
data/model_outputs/experiment_03/
data/model_outputs/experiment_04/
```

Each experiment contains outputs from multiple models, for example Gemini, Nvidia and Liquid.

## Stage 6: Summary and manual validation

After automatic evaluation, the project summarizes average scores, medians, counts of each score, and total evaluated rows.

Relevant code:

```bash
python -m src.analysis.summarize_classifications
```

The project can also create a random sample of 70 examples for manual scoring:

```bash
python -m src.analysis.create_manual_sample
```

