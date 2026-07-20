# Sarcasm Detection Fine-Tuning — Plan

This document summarizes the next phase of the project, first proposed at
the project's fourth supervisor meeting (2026-07-16) -- see
`docs/project_history.md` for the full meeting-by-meeting record.

This is a **planning document only** -- it describes what the fine-tuning phase
is meant to do and what data it will use. The actual training code and Azure VM
setup are a separate, later step, not covered here.

## Why this phase exists

The interpretation/neutralization work so far (see `methodology.md`
and `results.md`) assumed the input was already known to be
sarcastic. In practice, a recurring failure mode across all three generator
models (Gemini, Nvidia, Liquid) was that they often couldn't tell a sentence
was sarcastic at all, even when the system prompt said so explicitly (the
"literal trap" — see meeting 4 findings). Sarcasm **detection** turned out to
be a bottleneck that sits *before* neutralization, not something that can be
assumed away.

The plan is therefore to build a dedicated binary sarcasm classifier
(sarcastic vs. not) as a pre-processing filter, before attempting any
neutralization/translation.

## Target dataset: Sarcasm Corpus V2

Sarcasm Corpus V2 (UC Santa Cruz). Binary-labeled posts (`sarc` / `notsarc`),
split into three sarcasm categories. The raw files now live in
`data/raw/sarcasm_corpus_v2/` (three CSVs, columns: `class`, `id`, `text`):

| Category | File | Total posts | sarc | notsarc |
|---|---|---|---|---|
| General Sarcasm (GEN) | `GEN-sarc-notsarc.csv` | 6,520 | 3,260 | 3,260 |
| Hyperbole (HYP) | `HYP-sarc-notsarc.csv` | 1,164 | 582 | 582 |
| Rhetorical Questions (RQ) | `RQ-sarc-notsarc.csv` | 1,702 | 851 | 851 |

Each category is already perfectly balanced (50/50 sarc/notsarc).

## Planned pipeline

1. **Data preparation & tokenization**
   - Combine the three category files, keeping a `category` column so results
     can later be broken down by GEN/HYP/RQ separately (they are qualitatively
     different: e.g. rhetorical questions are a distinct linguistic pattern
     from general sarcasm).
   - Clean and deduplicate, then split into train/validation/test.
   - Tokenize with a BERT tokenizer to the model's expected input format.

2. **Model: BERT-based classification**
   - Fine-tune a pre-trained BERT encoder.
   - A linear classification layer is trained on top of the `[CLS]` token's
     hidden state (the aggregate sentence representation) to produce the
     final sarc/notsarc prediction.
   - This requires a GPU — this is the part that will run on the Azure VM.

3. **Evaluation**
   - Standard metrics: Accuracy and F1-score, likely broken down per category
     (GEN/HYP/RQ) given how different they are.
   - In-depth qualitative error analysis: manually reviewing misclassified
     examples to map out which linguistic/syntactic/contextual patterns are
     hardest for the model — the same kind of manual review already done for
     the interpretation outputs (see `results.md`).

## Not yet decided / open questions for the next planning session

- Exact train/val/test split ratios and whether to stratify by category as
  well as by label.
- Which pre-trained BERT checkpoint to start from (e.g. `bert-base-uncased`
  vs. a variant).
- Whether the three categories should be trained on jointly or evaluated
  separately as different difficulty tiers.
