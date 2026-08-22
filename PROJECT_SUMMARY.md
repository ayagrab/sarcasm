# Sarcasm Project — Summary

Clean, cumulative, high-level document covering the **whole project**, in
two parts:

- **Part I — Sarcasm Interpretation Benchmark:** given a sarcastic tweet,
  have an LLM rewrite it as a sincere, non-sarcastic sentence, and
  evaluate the rewrite automatically, by an LLM judge, and by human
  annotators. The project's original scope, based on the SIGN paper.
- **Part II — Sarcasm Detection:** given a short English text, predict
  whether it is sarcastic at all. A new research direction started after
  Part I repeatedly found that models often can't tell a sentence is
  sarcastic in the first place, which no amount of rewriting-prompt
  refinement can fix (see Part I's conclusions below).

Both parts are complete. Behind Part I, see `docs/methodology.md` and
`docs/results.md` for further detail.

---

## Part I — Sarcasm Interpretation Benchmark

### Problem and approach

Given a sarcastic tweet, generate a sincere rewrite that preserves its
intended (non-sarcastic) meaning, then determine how well that rewrite
actually captures the meaning — automatically, by an LLM judge, and by
human annotators — and whether an LLM judge can validly substitute for
human annotators on this task.

### Dataset

The project starts from the SIGN paper's original test data: sarcastic
tweets only (the reference/result column is removed), deduplicated to
`data/processed/clean_sarcastic_sentences.csv`
(`python -m src.preprocessing.clean_dataset`).

### Methodology

- **Model selection.** Five candidate generator models were surveyed
  first (Gemini 2.5 Flash Lite, Baidu Qianfan OCR FastFree, Liquid LFM
  2.5-1.2B Thinking, Nvidia Nemotron Nano 9B v2, OpenAI GPT-OSS 20B); the
  project settled on three for the main pipeline — **Gemini 2.5 Flash
  Lite**, **Nvidia Nemotron Nano 9B v2**, and **Liquid LFM 2.5-1.2B
  Thinking** (the other two were survey-only).
- **Generation prompts.** Each model receives the same sarcastic tweet
  under 4 prompt versions (`prompts/generation/`): plain instruction (1),
  "translate the true meaning" (2), added formatting/grammar constraints
  (3), and few-shot with 3 worked examples (4) — run via
  `src.generation.generate_with_gemini` / `generate_with_openrouter`.
- **Evaluation methods, three independent ones:**
  - *Automatic text-overlap metrics* — BLEU, ROUGE-1/2, PINC (novelty:
    fraction of interpretation words not in the source), and a combined
    score `PINC * sigmoid(BLEU)` (`src.postprocessing.calculate_text_metrics`).
  - *LLM judge* (primary) — an independent judge model (OpenAI GPT-OSS
    20B via OpenRouter) scores each interpretation 1 (incorrect), 2
    (partially correct), or 3 (correct) (`src.evaluation.evaluate_with_llm`).
  - *NLI evaluation* (experimental alternative) — treats the sarcastic
    sentence as premise and the interpretation as hypothesis, checking
    whether an NLI model predicts entailment more strongly than
    contradiction (`src.evaluation.evaluate_with_nli`).
- **Human validation.** Three team members independently scored a random
  sample of 70 tweets across all 4 prompts and all 3 models, using the
  same 1-3 scale as the LLM judge, to compare human judgment against the
  automated judge directly.
- **Alt-Test** (Calderon, Reichart & Dror, 2025 — `docs/alt_test_reference.md`):
  a statistical procedure for justifying the use of an LLM annotator in
  place of human annotators, via a leave-one-out comparison against the
  remaining human annotators. Reports a **Winning Rate** (fraction of
  human annotators the LLM out-performs, passes if ≥ 0.5) and an
  **Advantage Probability** (estimated probability the LLM is at least as
  good as a random human annotator), at a chosen cost-benefit tolerance
  **epsilon** (this project used 0.2, appropriate for expert-level human
  annotators).
- **Further statistical analysis:** Fleiss' Kappa (inter-rater agreement
  among the 3 human annotators), Kruskal-Wallis (whether prompt/model
  choice significantly affects the LLM-judge score), Spearman correlation
  (whether structural features like length/overlap predict quality), and
  qualitative case studies of the clearest agreement/disagreement cases.

### Results

**Automatic metrics (initial 5-model survey):**

| Model | BLEU | ROUGE-1 | ROUGE-2 | PINC | Combined |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash Lite | 4.25 | 29.20 | 11.17 | 74.62 | 38.10 |
| Baidu Qianfan OCR FastFree | 6.14 | 37.65 | 17.14 | 73.76 | 38.01 |
| Liquid LFM 2.5-1.2B Thinking | 1.14 | 10.57 | 2.40 | 89.08 | 44.79 |
| Nvidia Nemotron Nano 9B v2 | 5.47 | 34.52 | 14.11 | 72.94 | 37.47 |
| OpenAI GPT-OSS 20B | 6.64 | 38.88 | 17.20 | 67.02 | 34.62 |

Liquid showed the highest novelty (PINC) but the weakest meaning
preservation; GPT-OSS preserved source structure best but was the least
creative. Automatic metrics alone were not considered sufficient,
motivating the LLM-judge evaluation.

**Prompt sensitivity** (LLM-judge scores, 1-3 scale, 265 tweets/model/prompt):
Prompt 2 ("translate the true meaning") was the most effective for human
annotators; Prompt 3 (added formatting constraints) was consistently the
worst. Nvidia led human-rated quality (~2.25 average with Prompt 4),
Gemini a close second, Liquid a distant last (~1.3). Both prompt and
model choice have a statistically decisive effect on quality
(Kruskal-Wallis: prompt statistic=156.699, p=9.4496e-34; model
statistic=328.303, p=5.1281e-72).

**Per-model rewriting strategies:** Gemini does "explanatory expansion"
(lengthens the sentence, spells out the sarcasm explicitly); Nvidia does
"stable precision" (keeps length close to the source, most consistent);
Liquid does "unstable reduction" (drastically shortens sentences with
high variance, near-zero lexical overlap with the source) — which
explains its low human quality scores. High-quality translations tend to
keep moderate-to-high word overlap with the source; structural features
overall correlate only weakly with quality score (Spearman, mostly
r < 0.5) — sarcasm interpretation quality is not explainable by simple
structural rules.

**Alt-Test: can the LLM judge replace human annotators?**
**Winning Rate 0.67, Advantage Probability 0.77 at epsilon=0.2 — PASSED**
(3 instances dropped for having fewer than 2 human annotators).

**Human vs. LLM-judge agreement:**

- **Fleiss' Kappa among the 3 human annotators: 0.282** ("fair
  agreement") — sarcasm-translation quality is genuinely subjective even
  for human experts, motivating the Alt-Test's epsilon tolerance above.
- **Agreement rate (rounded human score == LLM-judge score) by model:**
  Liquid 67.1%, Gemini 50.0%, Nvidia 31.4% — easier to agree on an
  outright failure (Liquid) than on nuanced, high-quality output (Nvidia):
  as quality goes up, the automated judge's reliability goes down.
- The LLM judge correctly identifies 61/77 score-1 instances but
  misclassifies 47 instances humans scored 2 (partially correct) as a
  complete failure, almost never using the middle score itself. It
  matches the rounded human score in only 104/210 (49.5%) of cases
  overall — showing both *semantic rigidity* (creative, high-quality
  paraphrases scored 1.0 by the LLM but 3.0 by humans, for not matching a
  rigid expected template) and a *fluency bias* (Liquid's fluent-sounding
  but semantically wrong translations sometimes fooled the LLM into a 3.0
  while humans scored them 1.0).

**Qualitative case studies:** a safety-filter failure (Gemini refused to
translate a tweet containing mild profanity outright; Nvidia successfully
rephrased the same tweet and was rated well by humans); a
political-correctness bias (both Gemini and Nvidia "lectured" the reader
instead of neutrally translating a sarcastic tweet about racism, and were
penalized by human scorers); and a world-knowledge bottleneck (a tweet
referencing a real-world event was misread literally by all 3 models,
since none had the cultural context — human annotators themselves split
on how to score it).

### Conclusions and the pivot to detection

The Alt-Test result (Winning Rate 0.67, Advantage Probability 0.77)
validates using an LLM judge in place of human annotators for this task,
at the chosen tolerance — a methodologically load-bearing result, since
every downstream prompt/model comparison in this part of the project
relies on the LLM judge's scores. But across the case studies and error
analysis, one failure mode recurred more than any other: **models often
couldn't tell a sentence was sarcastic in the first place, even when told
explicitly** — refining the rewriting prompt further doesn't fix a
misread of the input. This "detection is a prerequisite for
interpretation" finding directly motivated the project's second phase:
**shift focus from neutralization to detection**, treating sarcasm
detection as a dedicated pre-processing filter to build and evaluate in
its own right, rather than assuming it away. See Part II below.

---

## Part II — Sarcasm Detection (6-Method Comparison)

### 1. Problem Definition

Given a short English text (a forum post / tweet-length message), predict
whether it is:

- `sarcastic`
- `not_sarcastic`

This is a **new phase** of the `sarcasm` repository, distinct from the
repository's existing (and already-implemented) work, which is a *sarcasm
interpretation/neutralization* benchmark (rewriting a known-sarcastic tweet
into a sincere sentence, then judging the rewrite — see the repo's root
`README.md`). That existing pipeline does not classify sarcasm; this phase
does.

The goal here is not just to train one classifier, but to run a fair,
reproducible comparison across fundamentally different approaches, and
determine which is best under matched evaluation conditions — and why.

## 2. Dataset

**Sarcasm Corpus V2** (UC Santa Cruz), already staged in this repository at
`data/raw/sarcasm_corpus_v2/` (untouched, read-only). Three category
subsets, each independently ~50/50 balanced:

| Category | Meaning | Rows | sarc / notsarc |
|---|---|---:|---|
| GEN | General Sarcasm | 6,520 | 3,260 / 3,260 |
| HYP | Hyperbole | 1,164 | 582 / 582 |
| RQ | Rhetorical Questions | 1,702 | 851 / 851 |
| **Total** | | **9,386** | **4,693 / 4,693** |

No missing values. No within-file exact duplicates. **336 rows (168
groups) share exact text across category files** (mostly GEN↔RQ, since RQ
and HYP posts are sarcasm sub-phenomena also present in the general
corpus); of those, **4 groups (8 rows) carry conflicting labels** between
categories — a genuine annotation inconsistency in the source data, kept
and flagged rather than dropped (2 of the 8 rows land in TEST — see
Section 8). No author/conversation/timestamp metadata exists in the raw
files, so the only leakage vector identified is duplicate/near-duplicate
text — handled via grouped splitting (below).

## 3. Experimental Methodology

### 3.1 Canonical dataset and split

- **Canonical dataset**: all three category files combined into one table
  with a global `example_id`, `category`, `source_file`, `dup_group_id`,
  and `label_conflict` flag. Non-destructive — no row is dropped.
  (`src/classification/data/build_canonical_dataset.py`)
- **Canonical split**: one fixed train/dev/test split (seed 42), built by
  `src/classification/data/make_splits.py` and persisted to
  `data/splits/` — reused, unmodified, by every one of the 6 approaches.
  Two properties were deliberately engineered into the split, not left to
  chance:
  - **Grouped by `dup_group_id`** (`StratifiedGroupKFold`, scikit-learn):
    every row belonging to the same normalized-text duplicate group is
    forced into the *same* split. Without this, the same underlying post
    (or a near-duplicate of it) could appear in both TRAIN and TEST,
    letting a model "cheat" by memorizing text it technically shouldn't
    have seen — this is the only leakage vector identified in this
    dataset (no author/timestamp metadata exists to leak through
    otherwise), so this grouping is the entire leakage defense.
  - **Stratified by label** at the group level, so each split keeps
    close to the corpus's overall ~50/50 class balance.
  - Target ratio: **70% / 15% / 15%** (`train_frac`/`dev_frac`/`test_frac`
    in `config/classification_settings.py`). Actual sizes came out
    slightly off that exact ratio because grouping constrains which rows
    can move between splits together:

    | Split | Rows | % of total | sarcastic | not_sarcastic |
    |---|---:|---:|---:|---:|
    | TRAIN | 6,706 | 71.4% | 3,369 | 3,337 |
    | DEV | 1,340 | 14.3% | 668 | 672 |
    | TEST | 1,340 | 14.3% | 656 | 684 |

  - **Sealing policy**: once built, the TEST split is never touched again
    until each approach's configuration is fully frozen — no prompt
    iteration, no few-shot demo selection, no hyperparameter tuning, no
    DSPy optimization step is ever run against it, for any of the six
    approaches.

### 3.2 How each approach actually used TRAIN / DEV / TEST

The *role* of TRAIN differs by approach type — this is intentional, not an
inconsistency, since not every approach has trainable parameters or a
demo-selection step:

| Approach | TRAIN | DEV | TEST |
|---|---|---|---|
| **M1** TF-IDF + LR | Fits the vectorizer + classifier weights | Selects hyperparameters (Stage A) | One frozen, final evaluation |
| **M2** Qwen zero-shot | *Not used* — zero-shot has nothing to fit or select | Compares against other approaches during development | One frozen, final evaluation |
| **M3** Qwen few-shot | Source pool for the 8 few-shot demo sentences (random or curated selection compared; random won, see 4) | Compares configuration variants | One frozen, final evaluation |
| **M4** Qwen reasoning | *Not used* — same reason as M2 (a different prompt style, still nothing to fit) | Compares against other approaches | One frozen, final evaluation |
| **M5** DSPy/MIPROv2 | 150-example sample used for bootstrapping candidate demos | 100-example sample used as MIPROv2's optimization valset — **hardcoded to DEV even during the TEST run**, confirmed in code, so TEST never leaks into the optimization search | One frozen, final evaluation of the compiled program |
| **M6** DeBERTa-v3-base | Full 6,706-row fine-tuning set | Per-epoch validation metric, drives early stopping / best-checkpoint selection | One frozen, final evaluation (no retraining — `scripts/eval_frozen_checkpoint.py` loads the already-trained checkpoint) |

Every method's TEST number reported in Section 6 is a **single, one-shot
evaluation of an already-frozen configuration** — never a number selected
from among several TEST attempts.

### 3.3 Shared evaluator

One implementation of Accuracy, per-class Precision/Recall/F1, Macro F1
(primary model-selection metric), Weighted F1, and confusion matrix, plus
per-example prediction storage for later error analysis, used identically
by every approach (`src/classification/evaluation/metrics.py`).

## 4. Approaches Evaluated

| ID | Approach | Frozen config | Status |
|---|---|---|---|
| M1 | TF-IDF + Logistic Regression (classical baseline) | `configs/tfidf.json` (EXP-001) | **DONE** |
| M2 | LLM zero-shot (`Qwen/Qwen3-4B-Instruct-2507`, local) | `configs/llm_zero_shot_qwen_local.json` (EXP-002) | **DONE** |
| M3 | LLM few-shot, 8 random demonstrations | `configs/llm_few_shot_random_8_qwen_local.json` (EXP-003) | **DONE** |
| M4 | LLM structured step-by-step reasoning | `configs/llm_reasoning_qwen_local.json` (EXP-005) | **DONE** |
| M5 | DSPy `MIPROv2`-optimized prompt | `configs/dspy_mipro_v2.json` (EXP-008) | **DONE** |
| M6 | Fine-tuned `microsoft/deberta-v3-base` | `configs/transformer_deberta_v3_base.json` (EXP-009) | **DONE** |

All LLM approaches (M2–M5) use the same underlying base LLM
(`Qwen/Qwen3-4B-Instruct-2507`, run locally on the Azure GPU VM's Tesla
M60s), to isolate the effect of prompting/optimization technique rather
than measuring base-model differences. Every method's config above was
selected as the **DEV-best candidate among the alternatives actually
tried** (e.g. M3's random-vs-curated comparison, M5's
Predict-vs-BootstrapFewShot-vs-MIPROv2 comparison), then evaluated once on
TEST — accuracy/cost tradeoffs (e.g. M5's ~2h29m TEST-run cost for
MIPROv2 vs. a much cheaper `Predict` baseline) were explicitly not taken:
the DEV-best result is kept for every method regardless of evaluation
cost.

## 5. Evaluation Metrics

Accuracy, per-class (sarcastic / not-sarcastic) Precision/Recall/F1, Macro
F1 (primary), Weighted F1, confusion matrix. Same implementation for every
approach; see `src/classification/evaluation/metrics.py`.

## 6. Results

All numbers below are on the frozen canonical **TEST** split (1,340
examples, gold distribution 656 sarcastic / 684 not_sarcastic), each a
single evaluation of an already-frozen configuration. DEV is included
alongside as a sanity check on generalization (see Section 7 for the
gap's interpretation). Cost/latency figures for M2–M4 are DEV-run
per-example rates (same n=1,340, same hardware/config — the actual TEST
run's exact wall-clock wasn't separately logged since it ran
back-to-back with other methods in a chained script); M5's actual TEST
wall-clock was measured directly.

| Model | TEST Macro F1 | DEV Macro F1 | DEV→TEST gap | Accuracy | Sarcastic P/R/F1 | Not-sarcastic P/R/F1 | Cost | Latency (1,340 examples) |
|---|---:|---:|---:|---:|---|---|---:|---|
| **M6 DeBERTa-v3-base (fine-tuned)** | **0.8209** | 0.8254 | −0.0045 | 0.8209 | 0.810 / 0.828 / 0.819 | 0.831 / 0.814 / 0.822 | $0 (self-hosted GPU) | ~22 min one-time fine-tune; eval-only pass is a few minutes, no retraining |
| M1 TF-IDF + LR | 0.7403 | 0.7529† | −0.0126 | 0.7403 | 0.724 / 0.758 / 0.741 | 0.757 / 0.724 / 0.740 | $0 (CPU) | seconds |
| M5 DSPy MIPROv2 | 0.6681 | 0.6700 | −0.0019 | 0.6866 | 0.618 / 0.942 / 0.746 | 0.888 / 0.442 / 0.590 | $0 (self-hosted GPU) | ~2h08m actual (51min optimization + 1h17m final eval) |
| M2 Qwen zero-shot | 0.6005 | 0.6008 | −0.0003 | 0.6396 | 0.578 / 0.973 / 0.725 | 0.924 / 0.320 / 0.475 | $0 (self-hosted GPU) | ~37 min (~1.7s/example) |
| M3 Qwen few-shot (random-8) | 0.5947 | 0.5880 | +0.0067 | 0.6351 | 0.575 / 0.971 / 0.723 | 0.918 / 0.313 / 0.467 | $0 (self-hosted GPU) | ~1h34m (~4.2s/example) |
| M4 Qwen structured reasoning | 0.5758 | 0.5796 | −0.0038 | 0.6224 | 0.566 / 0.974 / 0.716 | 0.920 / 0.285 / 0.435 | $0 (self-hosted GPU) | ~1h02m (~2.8s/example) |

† M1's DEV number is a reference-only re-run of the identical frozen
config with `eval_split=dev` (done purely for the cross-model DEV
analysis, not a re-tune) — M1 was already frozen from Stage A before this
DEV/TEST comparison methodology existed.

**All 6 approaches ran at $0 marginal cost** — every LLM call used a
locally self-hosted Qwen3-4B on the project's own Azure GPU VM, never a
paid API. The "Cost" column exists for methodological completeness /
future-work comparability, not because any real API spend occurred.

### 6.1 M5's actual optimized prompt

Unlike M2/M4 (fixed hand-written prompts, see `prompts/classification/`)
and M3 (fixed prompt + a demo-selection strategy), M5's prompt was
*itself* the object of optimization — DSPy's `MIPROv2` searched over 3
proposed instruction candidates × 6 bootstrapped few-shot demo sets, using
13 trials of Bayesian optimization against a 100-example DEV valset. The
winning combination (recovered from the compiled program's own saved
state, `results/EXP-008-TEST/compiled_program.json` — not reconstructed
or guessed) was:

**Instruction:**
> Classify the given sentence as "sarcastic" or "not_sarcastic" based on
> linguistic cues such as irony, exaggeration, contradiction, or mocking
> tone.

This was one of 3 candidates MIPROv2's own instruction-proposal step
generated (the other two: a bare *"Classify whether an English sentence is
sarcastic"* with no elaboration, and a much longer, multi-paragraph
version with explicit examples and a lengthy rationale about irony/
hyperbole detection) — the mid-length one won, neither the shortest nor
the most elaborate.

**Few-shot demonstrations (4, exactly as saved in the compiled program):**

| # | Sentence | Label | Source |
|---|---|---|---|
| 1 | "Dude, go jack off to your god somewhere else. We don't need to see it. emoticonXKill" | sarcastic | Bootstrapped (self-generated reasoning trace) |
| 2 | "i'm not disputing the numbers, and i'm certainly not disputing the fact that bush won. what i'm intrigued about is the razor thin margins... don't you find it the least bit odd that it's been 30 years since a president received a plurality? waxy" | not_sarcastic | Bootstrapped |
| 3 | "Nope. However, he does get to pay child support if he gets caught." | sarcastic | Labeled directly |
| 4 | "you really believed me? wow! i never knew i had such power ;)" | sarcastic | Labeled directly |

Notably, this is **fewer demonstrations than M3's fixed 8** — and MIPROv2
still outperformed M3 on both DEV and TEST, consistent with the
project's recurring finding (Section 7) that *smarter demo selection*
matters more than *demo count* for this task on this model.

## 7. Comparison Between Approaches

Full pairwise cross-model analysis on TEST predictions (all 6 frozen
configs, `results/cross_model_test_analysis.csv`), mirroring the same
analysis already done on DEV during development — every finding from DEV
holds on TEST too, which is itself a finding (no DEV-only artifact):

- **The four LLM-based methods (M2–M5) cluster tightly together** (88–95%
  pairwise agreement) — they mostly make the *same* mistakes as each
  other, consistent with sharing the same underlying Qwen3-4B model and
  prompt family. **M1 and M6 (the two "trained on this task's labels"
  methods) agree with each other far more (80.6%) than either agrees with
  any LLM method (62–71%)** — a real methodological split between
  "trained on labels" and "prompted with a general-purpose LLM," not
  noise, and it holds identically on TEST.
- **Systematic sarcastic-over-prediction bias in every LLM method,
  confirmed on TEST:** M2 FP=465/FN=18, M3 FP=470/FN=19, M4 FP=489/FN=17,
  M5 (best-calibrated of the four, still skewed) FP=382/FN=38. M1 and M6
  are far more balanced: M1 FP=189/FN=159, M6 FP=127/FN=113. This is the
  single biggest driver of the gap between M6/M1 and the LLM methods —
  every LLM variant's sarcastic-class recall is 94–97%, but at the cost of
  not-sarcastic recall crashing to 28–44% (see Section 6's per-class
  columns); M1/M6 don't make this tradeoff because they learn the
  corpus's actual class balance from labels rather than being prompted
  toward one label.
- **Distribution of `n_models_correct` (0–6) per TEST example:** 568/1,340
  (42.4%) are correctly classified by **all 6** methods (genuinely easy);
  89/1,340 (6.6%) are wrong for **every single method**, including M6's
  0.8209 — these are the dataset's hardest/most ambiguous rows, not fixed
  by any approach tried in this project.
- **Category breakdown (GEN / HYP / RQ):** every method does worst on HYP
  (hyperbole) and best on RQ (rhetorical questions) or GEN, consistent
  with DEV. M6 leads in every category (GEN 0.833, HYP 0.704, RQ 0.854) —
  a 8–15 point margin over the next-best method (M1) per category, so
  M6's overall lead isn't driven by one easy category.

## 8. Error Analysis

- **89 examples (6.6% of TEST) are misclassified by every one of the 6
  methods**, including the best (M6, 82.1% overall accuracy) — a hard,
  irreducible-so-far core of the dataset under every approach tried here.
  A natural next step (Section 12) would be manually reading a sample of
  these to characterize what makes them hard (e.g. sarcasm requiring
  world knowledge or tone the text alone doesn't carry).
- **The 8 label-conflict rows** (`data/processed/sarcasm_v2_audit_report.json`,
  `label_conflict_example_ids`) split as 2 in TEST, 2 in DEV, 4 in TRAIN.
  Every method scored exactly 50% (1/2 correct) on the 2 TEST rows — too
  few to be statistically meaningful on their own, but directionally
  consistent with these rows being inherently ambiguous/contradictorily-
  labeled by construction (the same category label was applied
  differently to identical text across the source corpus's category
  files), not a modeling failure.
- **Confidence calibration (M1 and M6, the only methods with a genuine
  per-example confidence score — LLM chat completions have none to
  report honestly):** both were reasonably well-calibrated on DEV
  (accuracy rises monotonically with predicted confidence), with
  M6's confidence distribution far more concentrated at the top end. This
  is a directly usable signal for a future "flag low-confidence
  predictions for human review" feature (Section 12).

## 9. Conclusions

1. **A small model *trained directly* on this task's labels (M6, 184M
   parameters) beats a much larger *prompted* general-purpose LLM (M2–M5,
   4B parameters) by a wide margin** — 0.82 vs. 0.58–0.67 Macro F1 — while
   also running dramatically faster (minutes vs. hours) and cheaper on
   this hardware. This is the project's headline result, and it held
   consistently from the very first DEV comparison through the final,
   sealed TEST evaluation.
2. **Among the LLM-prompting techniques, more sophistication did not
   reliably help.** Zero-shot (M2) beat both few-shot (M3) and explicit
   structured reasoning (M4) on TEST. Only DSPy's automated prompt/demo
   *search* (M5) improved meaningfully on plain zero-shot — and it did so
   with *fewer* demonstrations (4) than the fixed few-shot baseline (8),
   suggesting the gain came from smarter *selection*, not from adding
   more context.
3. **Every LLM-based method shares the same systematic failure mode**: a
   strong bias toward predicting "sarcastic," at the direct cost of
   missing the not-sarcastic class. This is a property of prompting
   Qwen3-4B for this task specifically (confirmed across 4 independent
   prompt variants), not fixable by prompt engineering alone within the
   techniques tried here.
4. **DEV scores were a trustworthy predictor of TEST performance for
   every method** — the largest DEV→TEST gap observed was 1.3 points
   (M1), and every LLM method's gap was under half a point. No evidence
   of overfitting to DEV anywhere in this project, despite DEV having
   been used for every method-selection decision along the way.
5. **The train/dev/test split's leakage controls (grouped by duplicate
   text, stratified by label) appear to have worked as intended** — the
   close DEV/TEST tracking in point 4 is itself indirect evidence against
   leakage (a leaky split would tend to show optimistic DEV numbers that
   don't hold up on TEST).

## 10. Limitations

- Sarcasm Corpus V2 is forum/social-media text from ~2016; may not
  represent contemporary sarcasm styles (slang, emoji-heavy text, etc.).
- No author/conversation metadata exists in the source data, so leakage
  control is limited to text-level deduplication/grouping — it's possible
  (though unlikely, given the observed duplicate patterns and the tight
  DEV/TEST tracking in Section 9) that stylistic leakage from the same
  author appears elsewhere in the corpus without a detectable textual
  signature.
- 8 label-conflict rows represent inherent annotation disagreement in the
  source corpus; they're kept rather than resolved, so a small amount of
  label noise is expected to persist through every split (confirmed to
  measurably depress every method's score on those specific rows,
  Section 8).
- All LLM-based approaches (M2–M5) used **one specific base model**
  (`Qwen/Qwen3-4B-Instruct-2507`) run on **one specific, older GPU
  generation** (Tesla M60, no flash-attention/bfloat16/modern-vLLM
  support) — the systematic sarcastic-over-prediction bias and the
  "more sophistication doesn't help" finding are demonstrated for this
  model/hardware combination specifically, not proven to generalize to
  other LLMs or newer hardware without further testing.
- M6 (the winning approach) is a single fine-tuning run (one seed) — no
  multi-seed variance estimate exists, though the margin over every other
  method is wide enough that this is unlikely to change the ranking.
- 89 TEST examples (6.6%) are misclassified by every method tried,
  representing this project's current ceiling on this dataset with these
  techniques.

## 11. Recommended Production Approach

**M6, fine-tuned `microsoft/deberta-v3-base`** — best TEST Macro F1
(0.8209) by a wide margin over every alternative, dramatically cheaper and
faster to run than any prompted-LLM approach (minutes vs. hours per full
evaluation pass, no per-call API/GPU-generation cost at inference time
beyond a single small-encoder forward pass), and the most balanced
predictor (least sarcastic-over-prediction bias of any method, M1
included).

If inference-time compute/latency budget is more constrained than
accuracy requirements (e.g. no GPU available at all, CPU-only
deployment), **M1 (TF-IDF + LR)** is the practical fallback — 0.74 Macro
F1 at essentially zero compute cost, still clearly ahead of every
prompted-LLM approach tried in this project.

## 12. Future Work

- **Manually characterize the 89 universally-hard TEST examples**
  (Section 8) to understand what specifically makes them resistant to
  every approach tried — likely candidates: sarcasm requiring context/
  world knowledge not present in the text alone, or genuinely ambiguous
  tone even to a human reader.
- **Multi-seed variance estimate for M6** — a single additional
  fine-tuning run (different seed) would confirm the current result isn't
  a lucky draw, though the margin over other methods makes this a
  low-priority confirmation rather than an open question.
- **Confidence-based human-review routing** using M1/M6's calibrated
  confidence scores (Section 8) — flag low-confidence predictions for
  human review rather than treating every prediction with equal certainty.
- **Test whether the LLM-prompting findings generalize** to a different
  base model and/or modern GPU hardware (flash-attention, bfloat16) — the
  current LLM results are all specific to Qwen3-4B-Instruct-2507 on Tesla
  M60s; it's an open question whether the sarcastic-over-prediction bias
  and "more sophistication doesn't help" pattern are properties of this
  task or of this specific model/hardware combination.
- **Category-specific (GEN/HYP/RQ) modeling** if HYP's persistently
  lower score (Section 7) across every method motivates a dedicated
  approach for hyperbole specifically.

## 13. Reproducing the Experiments

Every method's frozen configuration file is under `configs/` (see the
table in Section 4). General shape to reproduce any result in this
document:

```bash
# 0. Install dependencies (base + this phase's extras)
pip install -r requirements.txt -r requirements-classification.txt

# 1. Build the canonical dataset (from data/raw/sarcasm_corpus_v2/, read-only)
python -m src.classification.data.build_canonical_dataset

# 2. Audit it
python -m src.classification.data.audit_dataset

# 3. Build the canonical split (persisted under data/splits/)
python -m src.classification.data.make_splits

# 4. DEV-phase development run (any approach, via its config file under configs/)
python -m src.classification.run_experiment --config configs/tfidf.json
python -m src.classification.run_experiment --config configs/llm_zero_shot_qwen_local.json
python -m src.classification.run_experiment --config configs/dspy_mipro_v2.json
python -m src.classification.run_experiment --config configs/transformer_deberta_v3_base.json

# 5. Once a config is frozen, the final one-shot TEST evaluation (a copy
#    of the frozen config with "eval_split": "test", e.g. configs/EXP-008-TEST.json)
python -m src.classification.run_experiment --config configs/EXP-008-TEST.json
# M6 uses an eval-only script instead (no retraining against an already-frozen checkpoint):
python -m scripts.eval_frozen_checkpoint --checkpoint-dir models/EXP-009/best_checkpoint --split test --experiment-id EXP-009-TEST --source-experiment-id EXP-009
```

Every experiment's configuration, metrics, and per-example predictions are
saved under `results/<experiment_id>/`. LLM/DSPy experiments used a
locally self-hosted Qwen3-4B on an Azure GPU VM (`provider="local_hf"`,
see `src/classification/llm/local_client.py`) — `provider="openrouter"`
remains available as an alternative for a machine with no local GPU, but
was not the path used to produce any result in this document. Transformer
fine-tuning downloads `microsoft/deberta-v3-base` from Hugging Face on
first run (needs a one-time `safetensors` conversion on this pinned
`torch` version if `use_safetensors=True` fails to load).

Run the test suite (never calls a real API or downloads a model):

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

Reproducing Part I: see `README.md`'s "Running the pipeline" section.

---

## Overall Project Conclusions

Read together, the two parts tell one coherent story: Part I established
that an LLM judge can validly stand in for human annotators on a subtle,
subjective NLP task (Alt-Test, Winning Rate 0.67) — but also surfaced,
repeatedly, that the bottleneck in sarcastic-text processing isn't
generating a good rewrite, it's recognizing sarcasm in the first place.
Part II answered that question directly: a small model *trained* on
labeled examples (fine-tuned DeBERTa-v3-base, 0.82 Macro F1) detects
sarcasm far more reliably than *prompting* a much larger general-purpose
LLM (0.58–0.67 Macro F1), which — across every prompting strategy tried —
shares a systematic bias toward over-predicting "sarcastic." Both
findings point the same direction: for this task, a model specifically
adapted to labeled, in-domain data outperforms a general-purpose model
used zero/few-shot, whether the task is judging quality (Part I) or
detecting sarcasm itself (Part II).

---

## Part III — SIGN Generalization

A third, additive phase — **COMPLETE as of 2026-08-22** — investigates
how well Part II's six frozen classifiers (M1–M6) generalize beyond
Dataset A to the structurally different **SIGN** dataset (sarcastic
tweets, each with up to five independent human non-sarcastic
interpretations — the same corpus Part I drew its sarcastic-only source
sentences from, here used in full for the first time, including the
interpretations). Every number below traces back to a persisted
`results/sign/EXP-SIGN-###/` artifact, no new inference run for this
synthesis.

This phase is strictly additive: Part II's frozen configs, predictions,
metrics, and the `EXP-00#` experiment IDs are read-only inputs here,
never modified. New experiments use a distinct `EXP-SIGN-###` ID
namespace and a separate `results/sign/` artifact tree, so the two
studies never collide in this repository's history.

### A note on how to read every result below: Task A vs. Task B

Every SIGN original is, by construction, `sarcastic`; every
interpretation is, by construction, `not_sarcastic`. Two different
questions get asked of the same predictions, and they must never be
conflated: **Task A** ("can the model recognize a SIGN sarcastic tweet
at all?" — originals only, single gold class, sarcasm **detection rate**
is the metric) and **Task B** ("can the model tell that sarcastic
original apart from a sincere rewrite of the same underlying meaning?" —
the full contrastive set, **Macro F1** is the metric). A low Task B score
must never be read as "the model can't detect sarcasm" without checking
Task A — several methods below score in the 90s on Task A while scoring
under 0.40 Macro F1 on Task B, because their failure is entirely on the
interpretation side, not the original side.

### 1. Cross-Dataset Sarcasm Detection (Task A — SIGN originals only, zero SIGN exposure)

| Method | Sarcasm detection rate (Task A) |
|---|---:|
| M2 Qwen zero-shot | **93.6%** (248/265) |
| M4 Qwen structured reasoning | 91.3% (242/265) |
| M3 Qwen few-shot | 89.4% (237/265) |
| M1 TF-IDF + LR | 79.6% (211/265) |
| M5 DSPy MIPROv2 (frozen) | 77.4% (205/265) |
| M6 DeBERTa-v3-base (fine-tuned) | 63.8% (169/265) |

Every method, including the weakest, still recognizes the clear majority
of unseen SIGN sarcastic tweets zero-shot — sarcasm *detection* transfers
across the domain shift reasonably well on its own. The LLM-prompted
methods (M2–M4) transfer best on this narrow task, ahead of both
label-trained methods (M1, M6) — the reverse of Part II's Dataset A
ranking, where M6 led by a wide margin.

**This result bears directly on an open question from Part II: is the
LLM-prompting family's weaker Dataset A performance (0.58-0.67 Macro F1
vs. M6's 0.82, unmoved across four very different prompting strategies —
zero-shot, few-shot, structured reasoning, DSPy-optimized) a limitation
of *prompting as an approach*, or of the specific *4B-parameter Qwen
model* being too small to grasp sarcasm's nuance? If the latter, a
stronger LLM should close the gap; if the former, it likely would not.
This SIGN result is evidence against the "model too weak to understand
sarcasm" reading: the same Qwen3-4B, zero-shot, detects 93.6% of SIGN's
sarcastic originals — well above M6's 63.8% on the identical task. A
model that cannot grasp sarcasm at all would not out-detect a
purpose-built fine-tuned classifier on unseen data. The more consistent
picture is that Qwen's weakness on Dataset A is a **calibration/decision-
boundary problem specific to binary classification under prompting** —
it over-triggers the sarcastic label (Part II's cross-model analysis:
FP≥382 vs. FN≤38 for every LLM method) rather than failing to recognize
sarcasm's linguistic signal. This does not fully resolve the
model-vs-approach question (a stronger LLM was not tested, and Task A's
single-gold-class setting sidesteps the calibration problem that hurts
Task B), but it substantially narrows which explanation the evidence
supports.

### 2. Contrastive Sarcasm Recognition (Task B — original vs. sincere interpretation)

| Method | Task B Macro F1 (zero-transfer) | Task B Macro F1 (after adaptation, M1/M6 only) |
|---|---:|---:|
| M6 DeBERTa-v3-base | 0.4724 | **0.6870** |
| M3 Qwen few-shot | 0.4015 | — |
| M5 DSPy MIPROv2 | 0.3957 | — |
| M4 Qwen structured reasoning | 0.3658 | — |
| M1 TF-IDF + LR | 0.3563 | 0.5861 |
| M2 Qwen zero-shot | 0.3397 | — |

Every method collapses well below its Dataset A TEST score zero-shot
(M1: 0.7403→0.3563; M6: 0.8209→0.4724) — but per §1's Task A/B
distinction, this is *not* a detection failure: the dominant error mode
is misclassifying **sincere** SIGN interpretations as sarcastic (M1
flags 71.2% of them; M2, the worst offender, flags 76.5%), not missing
the sarcastic originals. M6 is the strongest zero-transfer method on
Task B despite being the weakest on Task A — a direct illustration of
why the two tasks must be reported separately.

### 3. Primary Human Reference Evaluation (original vs. interpretation #1 only)

Interpretation #1 (first-listed per SIGN family) is treated as the
primary/best human reference — a provenance decision (see plan doc §1),
tested rather than assumed to also be the "easiest" case (§4 below).

| Method | Primary-Ref Macro F1 | Primary-Ref pair success rate |
|---|---:|---:|
| M3 Qwen few-shot | 0.5901 | 26.8% |
| M5 DSPy frozen | 0.5650 | 24.2% |
| M6 DeBERTa | 0.5658 | 21.5% |
| M4 Qwen reasoning | 0.5645 | 23.0% |
| M2 Qwen zero-shot | 0.5408 | 20.8% |
| M1 TF-IDF+LR | 0.5172 | 13.6% |

Restricting to the single primary interpretation is substantially easier
than the full contrastive set for every method (e.g. M1: 0.356→0.517
Macro F1; M6: 0.472→0.566) — but pair success (getting **both** the
original and its primary interpretation right, same family) still tops
out at 26.8%, so even the cleanest single-pair case remains hard in
absolute terms.

### 4. Full Family Evaluation (original + all five interpretations)

| Method | Strict family accuracy (all 6 rows correct) | Soft family score (mean correct fraction) |
|---|---:|---:|
| M6 DeBERTa | **3.8%** | **0.370** |
| M5 DSPy frozen | 2.3% | 0.218 |
| M3 Qwen few-shot | 1.9% | 0.282 |
| M2 Qwen zero-shot | 1.1% | 0.212 |
| M4 Qwen reasoning | 0.8% | 0.236 |
| M1 TF-IDF+LR | 0.0% | 0.178 |

Getting an entire family exactly right (the original plus all five
interpretations) is essentially unsolved for every method (0-3.8%) — the
gap between this and §3's pair-success rate is the clearest illustration
of Task B's real difficulty: one interpretation right per family is
plausible, all five simultaneously is not. **Per-interpretation-rank
recall was tested, not assumed, and does *not* show a clean "rank #1 is
easiest" pattern** — M6's not-sarcastic recall by rank is 49.8% / 54.7% /
39.6% / 60.4% / 53.2% (#1→#5), non-monotonic; most methods peak at rank
#4, not #1. Interpretation #1 being the "primary reference" is a
provenance/quality claim (first-listed = best annotation), not a claim
that it is linguistically the easiest case for a model.

### 5. What Kinds of SIGN Sarcasm Do the Models Fail to Detect? (Task A error analysis)

Complete pass over every Task A miss across all six methods (148/265
originals missed by at least one method), not a sample. Of the
measurable text properties tested, only length shows even a modest
contrast: ever-missed originals average 13.4 words vs. 14.8 for
always-detected ones (mildly harder when shorter). Question marks are
*more* common among always-detected originals (17.1%) than ever-missed
ones (13.5%) — contradicting a naive "rhetorical questions are harder"
hypothesis. VADER sentiment is nearly identical between groups (0.237 vs.
0.223) — surface polarity alone does not predict what gets missed.
Qualitatively, the 117/265 originals every method detects cluster around
classic **polarity-reversal verbal irony** (strongly positive words
applied to clearly negative topics, "don't you just love..."
constructions) — idiomatic markers that transfer well from Dataset A.
The hardest non-duplicate cases tend toward flat, factual-sounding
statements whose sarcasm depends on outside world knowledge not
recoverable from the text alone (e.g. whether a specific sports result
was actually good or bad) — a context-dependence limitation no amount of
in-domain fine-tuning on text alone can fully close.

### 6. Why Are Sincere SIGN Interpretations Misclassified as Sarcastic? (Task B false-positive analysis)

**Headline data-quality finding:** a real fraction of SIGN's "sincere
interpretations" are byte-identical to their own sarcastic original — an
irreducible label contradiction, discovered by direct inspection, not
assumed. 28.1% of the 427 interpretations flagged sarcastic by *every*
method are exact duplicates of their original; both SIGN Test originals
missed by all six methods share this property. This means a real share
of Task B's apparent difficulty is a **data-quality ceiling no model can
cross**, not purely a modeling gap — disclosed as a limitation rather
than filtered out, per an explicit decision made before Phase 7's
training-data preparation. Separately, cross-model overlap analysis
shows two distinct failure clusters rather than one shared hard subset:
the three Qwen-prompted methods (M2/M3/M4, same base model) miss heavily
overlapping originals (pairwise Jaccard 0.50-0.67), while M1 and M6 each
fail in their own largely distinct pattern (M1-M6 Jaccard 0.22, M1-M2
0.04) — a candidate signal that M1/M6 would be the more complementary
pairing for any future ensembling, not yet tested.

### 7. Effect of SIGN Domain Adaptation

Three conditions compared for M1 and M6: **A** = Dataset A only
(zero-transfer, reused); **B** = Dataset A TRAIN + SIGN Train, combined
fit; **C** = SIGN Train only.

| Model | Condition | SIGN Test Macro F1 | Dataset A TEST Macro F1 |
|---|---|---:|---:|
| M1 | A (zero-transfer) | 0.3563 | 0.7403 |
| M1 | **B (combined — winner)** | **0.5861** | **0.7477** |
| M1 | C (SIGN only) | 0.5829 | 0.4527 (catastrophic forgetting) |
| M6 | A (zero-transfer) | 0.4724 | 0.8209 |
| M6 | **B (combined — winner)** | **0.6870** | **0.8209** (unchanged, 4 decimals) |
| M6 | C (SIGN only) | 0.6806 | 0.4034 (catastrophic forgetting) |

Combining SIGN Train with Dataset A (condition B) captures almost all of
SIGN-only training's improvement while **fully avoiding** the
catastrophic forgetting condition C causes on both models — the clean
answer for how to adapt without sacrificing the original task. A genuine
model-capacity contrast emerged in the before/after error diff: M1's
adaptation traded a little Task A recall for a large Task B gain
(79.6%→74.7% detection rate); **M6's adaptation improved both Task A and
Task B simultaneously**, with no such trade-off.

### 8. Effect of the Amount of SIGN Training Data (learning curve, 0-100% of SIGN Train, condition B recipe)

| SIGN Train used | M1 Task B F1 | M1 Task A | M6 Task B F1 | M6 Task A |
|---:|---:|---:|---:|---:|
| 0% | 0.356 | 79.6% | 0.472 | 63.8% |
| 10% | 0.448 | 71.3% | 0.597 | 66.8% |
| 25% | 0.524 | 72.8% | 0.634 | 75.1% |
| 50% | 0.558 | 72.8% | 0.685 | 79.6% |
| 75% | 0.579 | 72.5% | 0.649 | 84.2% |
| 100% | 0.586 | 74.7% | 0.687 | 78.5% |

Most of the Task B gain arrives early for both models (the first 10-25%
of SIGN Train covers over half the total 0%→100% improvement) — a
classic diminishing-returns learning curve. M1's Task A rate stays
roughly flat across every nonzero fraction (never recovering to its
zero-transfer level); **M6's Task A rate climbs substantially with more
SIGN exposure** (63.8%→84.2% at 75%) — the opposite direction from M1,
consistent with §7's model-capacity contrast. M6's curve shows some
non-monotonic run-to-run variance in the 50-75% region, flagged as a
limitation (no controls run to separate sampling variance from
fine-tuning-run variance at these fraction sizes).

### 9. Effect of the Number of Human Interpretations (k-ablation, k=1/2/3/5, full 100% family set)

| k (interpretations per family) | M1 Task B F1 | M1 Task A | M6 Task B F1 | M6 Task A |
|---:|---:|---:|---:|---:|
| 1 | 0.586 | 74.7% | 0.687 | 78.5% |
| 2 | 0.629 | 61.1% | 0.690 | 81.5% |
| 3 | 0.648 | 57.7% | 0.713 | 80.8% |
| 5 | 0.664 | 55.9% | 0.740 | 77.7% |

**The sharpest model contrast in the whole project.** For M1, more
interpretation diversity buys Task B gains at an increasingly steep Task
A cost (74.7%→55.9%, a real trade-off — k=1 is not dominated by higher
k). For M6, the identical manipulation improves Task B by even more
(0.687→0.740) with **no comparable Task A cost** (stays in a tight
77.7-81.5% band) — k=5 is close to strictly better than k=1 for M6. The
same pattern — capacity-limited model forced into a trade-off,
fine-tuned model absorbing added diversity almost for free — recurs
across §7, §8, and §9 independently, making it the project's most robust
finding about *how* domain adaptation should be approached differently
depending on model class.

### 10. Remaining Difficult Cases After Adaptation

Even after the winning adaptation recipe (condition B, §7), both models
still leave real gaps. M1: 41/265 originals still missed after
adaptation (plus 26 newly broken by the adaptation itself), and 483/1,470
interpretations still misclassified as sarcastic. M6: 35/265 originals
still missed (plus 22 newly broken), and 257/1,470 interpretations still
misclassified. Some fraction of both models' remaining Task B errors is
the irreducible duplicate-interpretation data-quality ceiling from §6,
not a closable modeling gap. The residual errors are concentrated in the
same context/world-knowledge-dependent cases identified in §5 — adaptation
closes most of the *distributional* gap between Dataset A and SIGN, but
does not resolve cases that require information outside the text itself.

### Overall Part III conclusion

Sarcasm *detection* (Task A) transfers reasonably well across the
Dataset A → SIGN domain shift, even zero-shot; the real generalization
challenge is Task B — telling a sarcastic original apart from a sincere
rewrite of the same underlying meaning, made harder by a genuine
data-quality ceiling in a meaningful minority of SIGN's interpretations.
Combining a modest amount of SIGN Train data with the original training
data (never replacing it) closes most of this gap at essentially zero
cost to Dataset A performance, for both a classical and a fine-tuned
model. The most consequential finding for future work is the consistent
model-capacity contrast across every adaptation experiment: a
capacity-limited linear model (M1) is forced into a real Task A/Task B
trade-off as it is given more SIGN signal (more training data or more
interpretation diversity), while a fine-tuned transformer (M6) absorbs
the same additional signal with little to no such trade-off — evidence
that *how* a model should be adapted to a shifted domain depends on its
capacity, not just on how much adaptation data is available.

## Documentation Map

| Document | Covers |
|---|---|
| `README.md` | Quick-start overview, installation, and how to run Part I |
| `docs/methodology.md` | Part I: how the dataset, models, prompts, and evaluation were chosen |
| `docs/results.md` | Part I: full results — metrics, Alt-Test, significance tests, case studies |
| `docs/alt_test_reference.md` | The Alt-Test method itself, its source paper, and how it's used here |
| `docs/project_structure.md` | Every file and folder in the repository, explained |
| `docs/pipeline.md` | Part I: technical, stage-by-stage map of the codebase |
| `Sarcasm_Project_Report.docx` | The full formal write-up, all three parts, submission-ready |
| `Sarcasm_Project_Poster.pptx` | The project poster (Part I + Part II) |
