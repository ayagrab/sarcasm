# Meeting Notes Summary — Full Record

This document consolidates everything presented across all four project
meeting decks in `docs/meeting_slides/`, so you don't need to open the
`.pptx` files to remember what was covered. It's organized chronologically
by meeting date.

Note on numbering: the title slides of the last three decks all say
"Meeting 3" — this looks like the students reused a template without
updating the label rather than an actual repeat of meeting 3. The dates
(embedded in each file name) are unambiguous and are what this document
uses to order things. Group: Anat, Sofia, Aya, Yehoraz. Supervisor: Lotem.

---

## Meeting — 2026-05-28 (file labeled "Meeting 2")

**Starting point: the original SIGN paper.** The paper introduces "sarcasm
interpretation" — converting a sarcastic sentence into a clear, non-sarcastic
statement that keeps the original meaning and sentiment. It built a dataset
of 3,000 sarcastic tweets and tested machine-translation approaches, and
proposed a model called **SIGN** that focuses especially on sentiment-bearing
words (since sentiment words are central to how sarcasm works). Automatic
metrics scored all models similarly, but human evaluators preferred SIGN's
outputs because they preserved meaning and sentiment more accurately.

**What was agreed in the first (undocumented) meeting with Lotem:** start by
surveying different AI models on the market and comparing their results on
the same test set/metrics used in the original paper. Models were chosen to
be as diverse as possible (open-source vs. proprietary, country of origin,
architecture, etc.).

**The 5 models tested:**

| Model | Developer | Country | Open source? | Params | Notes |
|---|---|---|---|---|---|
| Gemini 2.5 Flash Lite | Google | USA | No | undisclosed (lightweight) | ultra-low latency, multimodal |
| Baidu Qianfan OCR FastFree | Baidu | China | No | undisclosed | specialized for OCR, bilingual CN/EN |
| Liquid LFM 2.5-1.2B Thinking | Liquid AI | USA | Yes (open weights) | 1.2B | non-transformer architecture, built-in "thinking" |
| Nvidia Nemotron Nano 9B v2 | NVIDIA | USA | Yes (open weights) | 9B | on-device, tool-calling / RAG focused |
| OpenAI GPT-OSS 20B | OpenAI | USA | Yes (open source) | 20B | open, transparent baseline |

**Method:** from the paper's git repo, kept only the test tweets (dropped
the reference/results column). Sent each tweet to every model with:

> Task: Interpret the following sarcastic tweet.
> Goal: Write one natural, non-sarcastic sentence that preserves the
> intended meaning.
> Rules: Output only the interpretation. Do not explain. Do not mention
> sarcasm.

This is `prompts/generation/generation_prompt_v1.txt` (Prompt 1).

**Metrics used** (this vocabulary is used throughout the whole project):
- **BLEU** (precision-oriented): rewards outputs that reuse the reference's
  exact wording. Low BLEU = very different phrasing from the reference.
- **ROUGE-1 / ROUGE-2** (recall-oriented): how much of the reference is
  covered by the output. ROUGE-1 = single-word overlap, ROUGE-2 = word-pair
  overlap (stricter).
- **PINC** (novelty-oriented): how different the output is from the
  reference — a high score means heavy paraphrasing/less copying.
- **Combined score** = `PINC * sigmoid(BLEU)` — rewards outputs that are
  both novel *and* still aligned in meaning.

**Results:**

| Model | BLEU | ROUGE-1 | ROUGE-2 | PINC | Combined |
|---|---|---|---|---|---|
| Gemini 2.5 Flash Lite | 4.25 | 29.20 | 11.17 | 74.62 | 38.10 |
| Baidu Qianfan OCR FastFree | 6.14 | 37.65 | 17.14 | 73.76 | 38.01 |
| Liquid LFM 2.5-1.2B Thinking | 1.14 | 10.57 | 2.40 | 89.08 | 44.79 |
| Nvidia Nemotron Nano 9B v2 | 5.47 | 34.52 | 14.11 | 72.94 | 37.47 |
| OpenAI GPT-OSS 20B | 6.64 | 38.88 | 17.20 | 67.02 | 34.62 |

**Interpretation:** Liquid's non-transformer architecture produced the
highest novelty (PINC) but the worst content preservation — likely unsuited
for further work. GPT-OSS scored highest on BLEU/ROUGE (best at preserving
original structure/meaning) but lowest PINC (least creative, sticks to
familiar patterns) — a good candidate if the goal is minimal-change
translation.

**Idea floated for future work:** sarcasm-focused fine-tuning of the GPT
model, since it performed best and the team has access to the original
paper's training data. (This direction was later reconsidered — see the
2026-07-16 meeting, where fine-tuning shifted from GPT/interpretation to a
BERT-based sarcasm **detector**.)

---

## Meeting — 2026-06-16 (file labeled "Meeting 3")

**Recap:** the same 5 models were tested with Prompt 1.

**Validation strategy introduced:** to check whether the sarcasm was
accurately captured, an independent model acts as a judge. If a success
threshold is met across all models, the team proceeds to the next phase with
Lotem; otherwise, they iterate on prompt engineering.

**Architecture adopted:** narrowed down to **3 generator LLMs** (Gemini
2.5 Flash Lite, Liquid LFM 2.5-1.2B Thinking, Nvidia Nemotron Nano 9B v2 —
Baidu and GPT-OSS dropped from the generator role) plus a **4th LLM
(ChatGPT) as an independent judge**.

**The judge prompt** (`prompts/evaluation/llm_judge_prompt.txt`) scores each
interpretation on a strict 1–3 scale:
- **1 = Incorrect** — wrong, literal, unrelated, or misses the sarcasm.
- **2 = Partially correct** — gets the general idea but misses context/tone/
  nuance/details.
- **3 = Correct** — accurately captures the intended sarcastic meaning.

The prompt explicitly instructs the judge to use score 2 frequently and only
give a 3 when clearly accurate ("if unsure between 2 and 3, choose 2").

**Prompt sensitivity results** (average / median / count of each score,
out of 265 tweets per model per prompt):

| Prompt | Model | Average | Median | #1 | #2 | #3 |
|---|---|---|---|---|---|---|
| 1 | gemini-2.5-flash-lite | 1.672 | 1 | 164 | 24 | 77 |
| 1 | liquid-lfm-2.5-1.2b-thinking | 1.570 | 1 | 164 | 51 | 50 |
| 1 | nvidia-nemotron-nano-9b-v2 | 2.079 | 3 | 117 | 10 | 138 |
| 2 | gemini-2.5-flash-lite | 2.230 | 3 | 95 | 14 | 156 |
| 2 | liquid-lfm-2.5-1.2b-thinking | 1.351 | 1 | 202 | 33 | 30 |
| 2 | nvidia-nemotron-nano-9b-v2 | 2.185 | 3 | 98 | 20 | 147 |
| 3 | gemini-2.5-flash-lite | 1.475 | 1 | 196 | 12 | 57 |
| 3 | liquid-lfm-2.5-1.2b-thinking | 1.106 | 1 | 245 | 12 | 8 |
| 3 | nvidia-nemotron-nano-9b-v2 | 1.660 | 1 | 173 | 9 | 83 |

Prompt 2 (`generation_prompt_v2.txt`, "translate the true meaning") was the
best-performing prompt at this stage; Prompt 3 (`generation_prompt_v3.txt`,
emphasis on spacing/grammar) was consistently the worst.

**Examples of bad translations (score 1)** — the recurring failure was
models translating sarcasm *literally*, i.e. missing that it was sarcastic
at all: e.g. "4 more assignments wooo college is so fun" → Gemini rendered
it as genuine enthusiasm about academic progress, when the intended meaning
is the opposite (frustration).

**Examples of good translations (score 3)** — e.g. "ahhhh property taxes i
love june" → Gemini/Prompt 2: "June is a terrible month because of property
taxes" — correctly inverts the sarcastic praise into the real complaint.

---

## Meeting — 2026-07-09 (file labeled "Meeting 3", draft of the next meeting)

**Recap:** GPT-OSS 20B's judgments (on Gemini/Liquid/Nvidia outputs) were
presented. Conclusion: the team needs to check whether the LLM judge can be
*trusted* by comparing it against human annotators — i.e. run an **Alt-Test**
— and also try a new prompt with in-context examples (few-shot) to see if it
improves quality.

**Prompt 4 introduced** (few-shot, `generation_prompt_v4.txt`) — same task as
before, but with 3 worked examples of sarcastic tweet → correct direct
translation included in the prompt itself.

**Alt-Test data setup:**
- 70 translations randomly sampled from the outputs of all 4 prompts.
- Evaluated by 3 human annotators.
- LLM evaluation dataset: 210 annotations (70 × 3 LLMs).
- Human evaluation dataset: 630 annotations (70 × 3 LLMs × 3 human
  annotators).
- **ε (epsilon) = 0.2**: sarcasm interpretation is inherently subjective, so
  a wider tolerance band is allowed when deciding if a score "matches".
- **Metric: neg_rmse** (negative root-mean-squared-error) — chosen because
  the scores are numeric (1.0/2.0/3.0), so RMSE captures the *magnitude* of
  disagreement, not just exact/inexact match.

**Alt-Test code sketch shown:**
```python
import json
with open('humans_annotations.json', 'r', encoding='utf-8') as f:
    humans_data = json.load(f)
with open('llm_annotations.json', 'r', encoding='utf-8') as f:
    llm_data = json.load(f)

scoring_metric = 'neg_rmse'
epsilon_value = 0.2

winning_rate, advantage_prob = alt_test(
    llm_annotations=llm_data,
    humans_annotations=humans_data,
    scoring_function=scoring_metric,
    epsilon=epsilon_value,
    min_instances_per_human=50
)

if winning_rate >= 0.5:
    print("Passed - the LLM can replace humans for this task.")
else:
    print("Failed - the LLM is not yet good enough to replace humans.")
```

**First Alt-Test result (draft):** 3 instances dropped (fewer than 2
annotators). **Winning Rate: 0.67**, **Advantage Probability: 0.77** →
the LLM can replace humans on this task (this result was later revisited
and calibrated further — see the 2026-07-16 meeting).

---

## Meeting — 2026-07-16 (file labeled "Meeting 3", final/expanded version)

This is the most detailed deck (28 slides) and supersedes the 07-09 draft.
It covers: a recap, the Alt-Test finalized, deep manual error analysis, and
the plan for the next phase (BERT fine-tuning for sarcasm **detection**).

### Recap of everything up to this point
- Meeting 1: evaluated 5 models with automatic metrics (BLEU/ROUGE/PINC/
  Combined).
- Meeting 2: built the 3-generator + 1-judge architecture, ran a prompt
  sensitivity analysis across 3 prompts, and started mapping edge cases.

### The 4 prompts (final recap)
1. **Prompt 1** — plain instruction, no examples.
2. **Prompt 2** — "translate the true meaning," output only the sentence.
3. **Prompt 3** — adds formatting/grammar constraints (spacing, one
   grammatical sentence).
4. **Prompt 4** — few-shot, with 3 worked examples.

### Action items that came out of the previous meeting
- **LLM Judge correlation check**: sample 40–50 tweets, have every team
  member manually grade every model's translation, and check alignment with
  the LLM judge.
- **Dataset cleaning**: flag tweets that aren't actually sarcastic; remove by
  majority consensus.
- **Few-shot exploration**: design/evaluate Prompt 4 and check whether
  in-context examples improve accuracy and nuance.

### Human validation & annotation process
- 3 team members independently graded a sample of **70 tweets** (see
  `data/manual_scoring/`: `random_70_for_manual_scoring.numbers` is the
  master sample; `anat.numbers`, `aya.numbers`, `yehoraz.numbers` are each
  annotator's completed scores).
- Each row: the original sarcastic tweet + the 3 models' translations, across
  all 4 prompt variations.
- Same 1–3 scale as the LLM judge, specifically so human vs. judge agreement
  could be measured directly.

### Challenges/findings while labeling by hand
- **The sarcasm blind spot**: models very often didn't recognize sarcasm at
  all, despite explicit system instructions — falling into the "literal
  trap" (e.g. "best long weekend ever" translated as a genuinely happy
  event; "another night of work oh the joy" translated as looking forward to
  work).
- **Subjectivity in real time**: annotators repeatedly debated the boundary
  between "partially correct" (2) and "fully correct" (3), well before any
  formal statistics (Fleiss' Kappa) were run — confirming sarcasm
  neutralization has no single ground-truth answer.
- **"Robot" translation fatigue**: Liquid LFM was fatiguing to grade because
  it often produced abstract, disconnected diagnostic statements instead of
  a translation (e.g. a complaint about a toddler → "The statement
  underscores triviality"; a complaint about credit-taking → "Acknowledgments
  merit proper recognition").
- **Over-sensitized safety filters**: commercial models (especially Gemini)
  were easily triggered by mild profanity (e.g. "bitch"), refusing to
  translate at all instead of producing an interpretation — a real
  limitation for real-world sarcastic text, which often contains informal
  language.

### Alt-Test — methodology & calibration (finalized)
- Same setup as the 07-09 draft: 70 translations, 3 LLMs, 3 human
  annotators. LLM dataset: 210 annotations. Human dataset: 630 annotations.
- **ε = 0.2**, metric = **neg_rmse** (same reasoning as before).

### Alt-Test — results (finalized)
- **Winning Rate: 0.67** — the LLM aligns better with the remaining human
  annotators than an excluded human, in the majority of comparisons.
- **Advantage Probability (ρ): 0.77** — very strong evidence the LLM
  represents the group consensus better than any individual human.
- **Epsilon calibration issue found**: at ε=0.15, the test gave a
  *self-contradictory* result — Winning Rate dropped to 0.33 (fail, below
  0.5) while Advantage Probability stayed high at 0.77 (implying the LLM is
  at least as good as humans). This contradiction is exactly why ε was
  raised to 0.20 — to properly account for sarcasm's inherent subjectivity.
- **Verdict: PASSED.** Core conclusion: it is statistically justified to
  replace human annotators with the LLM judge for this specific task.

### Statistical significance & human consistency
- **Fleiss' Kappa among human annotators: 0.282** ("fair agreement") —
  confirms sarcasm interpretation is genuinely subjective, even among
  human experts. This is what justifies using a *positive* tolerance (ε=0.2)
  in the Alt-Test rather than requiring exact score matches.
- **Kruskal-Wallis tests**: both prompt choice and model choice have a
  statistically decisive impact on quality:
  - Prompt effect: p = 9.45×10⁻³⁴
  - Model effect: p = 5.13×10⁻⁷²

### Model performance & rewriting strategies
- **Human-rated averages**: Nvidia led (~2.25), Gemini close second, Liquid
  failed significantly (~1.3).
- **Distinct rewriting behaviors observed:**
  - **Gemini** — "explanatory expansion": tends to lengthen the sentence,
    spelling out and resolving the sarcasm explicitly.
  - **Nvidia** — "stable precision": keeps a highly consistent sentence
    length close to the source.
  - **Liquid** — "unstable reduction": drastically shortens sentences, with
    high structural variance.
- **Prompt 2** was the most effective prompt for human annotators; **Prompt
  3** was consistently worst.
- **Prompt 4 (few-shot)** was the *only* configuration where the LLM judge
  (ChatGPT) scored translations *higher* than humans did — a sign of machine
  bias toward structured, in-context-example-matching output.

### Human vs. machine — evaluation discrepancies & biases
- **Failure detection is easier than quality evaluation.** High agreement
  (67.1%) on Liquid's outright failures (both humans and ChatGPT easily spot
  literal translations that are obviously wrong). Low agreement (31.4%) on
  Nvidia — the *best*-performing model — because as quality/nuance goes up,
  the automated judge starts penalizing creative (but correct) paraphrases.
- **Semantic rigidity**: ChatGPT gave a 1.0 to some high-quality, semantically
  correct paraphrases from Nvidia/Gemini that humans scored 3.0, because they
  didn't match a rigid expected template.
- **Fluency bias**: ChatGPT gave a perfect 3.0 to some of Liquid's *failed*,
  literal translations (e.g. "college is so fun" translated as genuinely
  positive) simply because the output sounded fluent/"academic".
- **Severity bias**: ChatGPT misclassified 47 examples that humans scored 2
  (partially correct) as a full failure (1), and rarely used the middle
  score at all.
- Structural text metrics (sentence length, etc.) correlate weakly (mostly
  <0.5, and Spearman r<0.20 in the deepest analysis) with human quality
  scores — translation quality is not explainable by simple structural
  features; it requires semantic understanding.
- ChatGPT's score matched the rounded human score only about half the time,
  most often under-rating mid-quality (2) translations as low quality (1).

### NLP metrics & linguistic anchors
- **Creativity vs. fidelity trade-off**: from Prompt 1→4, BLEU/ROUGE
  (similarity to source) increased while PINC (novelty) decreased. Prompt 4
  (few-shot) made models the most literal/closest to source structure.
- **Word overlap matters, but only up to a point**: high-quality translations
  (human score 3.0) kept a moderate-to-high median word overlap (~30%) with
  the source. Extremely aggressive rewriting (Liquid, ~85% PINC, near-zero
  overlap) lost the original meaning entirely.
- **Core conclusion**: sarcasm interpretation is deeply semantic and cannot
  be predicted or evaluated by structural rules alone (length, overlap,
  etc.) — this is exactly why intelligent evaluators (human or LLM) are
  necessary, and also motivates moving toward a learned classifier rather
  than more rule/prompt engineering.

### Manual data review — case studies
- **Safety alignment failure (Gemini)**: a tweet using mild profanity
  ("bitch") got a hard refusal from Gemini ("I cannot fulfill this
  request..."). Nvidia, on the same tweet, successfully rephrased the
  vulgar sarcasm into a polite request and was rated well by humans (scores
  of 2–3). This shows commercial RLHF safety alignment actively hinders
  real-world NLP use on informal/profane text.
- **Political-correctness / preachiness bias**: on a tweet about "sending
  racists to Africa," both Gemini and Nvidia turned the sarcasm into an
  ideological/educational lecture instead of a neutral translation — both
  penalized by human scorers (1–2).
- **Fluency illusion**: Liquid failed to detect sarcasm in "college is so
  fun" (translated as sincerely positive) but phrased it in polished,
  academic-sounding language ("The statement underscores triviality"),
  fooling ChatGPT into a perfect 3.0 while humans unanimously scored it 1.0.
- **World-knowledge bottleneck**: a tweet asking (sarcastically) for photos
  of "Ali with the Beatles" was posted the day Muhammad Ali died, when the
  internet was flooded with that exact photo — the tweet is really a
  complaint about repetitive content. All 3 models missed this real-world
  context and interpreted it as a literal, sincere request. Humans
  themselves split on how to score it (1 vs. 3), depending on whether they
  recognized the context.
- **Inherent human subjectivity example**: "being alone is always fun" —
  Liquid's interpretation ("The author perceives solitude as unenjoyable")
  got both a 3 and a 2 from different human annotators on the same
  translation — consistent with the 0.282 Fleiss' Kappa finding.

### Next steps (decided at this meeting)
- **Shift focus from neutralization to detection.** The recurring
  bottleneck across the whole project was that models often can't tell a
  sentence is sarcastic in the first place — even when told explicitly.
  Continuing to refine neutralization prompts doesn't fix that.
- **Plan: fine-tune a dedicated BERT-based binary sarcasm classifier**
  (sarcastic vs. non-sarcastic), to serve as a pre-processing filter *before*
  attempting any neutralization/translation.
- **Target dataset**: Sarcasm Corpus V2 (UC Santa Cruz) — General Sarcasm
  (GEN, 6,520 posts), Hyperbole (HYP, 1,164 posts), Rhetorical Questions
  (RQ, 1,702 posts), each labeled `sarc`/`notsarc`.
- **Planned approach**: clean/tokenize the corpus into train/val/test splits;
  fine-tune a pre-trained BERT encoder with a linear classification layer on
  top of the `[CLS]` token; evaluate with Accuracy/F1; do an in-depth
  qualitative error analysis on misclassifications (same spirit as the
  manual review done for the interpretation task).
- See `docs/finetuning_plan.md` for the write-up of this plan, and
  `data/raw/sarcasm_corpus_v2/` for the raw corpus files staged for it.
