# Sarcasm Detection — Project Summary

Clean, cumulative, high-level document. For the detailed technical record
(exact commands run, environment audit, per-method debugging) see
`EXPERIMENT_LOG.md`.

**Status: Stage B is complete.** All 6
approaches (M1–M6) have been developed on DEV, frozen to a single final
configuration each, and evaluated exactly once on the sealed TEST split
(1,340 examples, never touched before freezing). **Fine-tuned
`microsoft/deberta-v3-base` (M6) is the best approach by a wide margin —
TEST Macro F1 0.8209** — outperforming every prompted-LLM approach (M2–M5,
all in the 0.57–0.67 range) and the classical TF-IDF+LR baseline (M1,
0.7403). Full results, methodology, and analysis below.

The VM was restarted **nine times** in total across this project's
runtime (a mix of deliberate and unannounced/cause-unknown restarts),
wiping the ephemeral `/mnt` disk every time — recovered fully every time,
with the recovery procedure becoming a proven, repeatable runbook (see
`EXPERIMENT_LOG.md`). The most costly incident was an unplanned outage
~35 minutes into M5's final TEST run, which cost a full restart of that
~2h+ optimization (no partial-checkpoint capability) — everything else
survived because results/configs are committed to git immediately after
each experiment finishes, never held only on the VM's ephemeral disk.

## 1. Problem Definition

Given a short English text (a forum post / tweet-length message), predict
whether it is:

- `sarcastic`
- `not_sarcastic`

This is a **new phase** of the `sarcasm` repository, distinct from the
repository's existing (and already-implemented) work, which is a *sarcasm
interpretation/neutralization* benchmark (rewriting a known-sarcastic tweet
into a sincere sentence, then judging the rewrite — see the repo's root
`README.md`). That existing pipeline does not classify sarcasm; this phase
does. See `EXPERIMENT_LOG.md`'s "Overview" section for the full detail.

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
text — handled via grouped splitting (below). Full detail:
`EXPERIMENT_LOG.md`'s "Dataset" section.

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
    DSPy optimization step is ever run against it. `EXPERIMENT_LOG.md`
    documents, per experiment, that this was followed.

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
tried** (see each method's section in `EXPERIMENT_LOG.md` for the
full reasoning, e.g. M3's random-vs-curated comparison, M5's
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
analysis already done on DEV during development (see
`EXPERIMENT_LOG.md`'s "Cross-Model Analysis" section) — every finding
from DEV holds on TEST too, which is itself a finding (no DEV-only
artifact):

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
  (accuracy rises monotonically with predicted confidence; see
  `EXPERIMENT_LOG.md`'s Cross-Model Analysis section for the exact bins), with
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

See `EXPERIMENT_LOG.md`'s per-method sections (M1 through M6) for the
exact command and configuration used for every result in this document.
General shape:

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
`torch` version — see `EXPERIMENT_LOG.md`'s M6 section for the exact
snippet if `use_safetensors=True` fails to load).

Run the test suite (never calls a real API or downloads a model):

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
