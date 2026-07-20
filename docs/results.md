# Results

What was found, using the methods described in `methodology.md`. For the
full chronological narrative, see `project_history.md`.

## Automatic metrics (initial 5-model survey)

| Model | BLEU | ROUGE-1 | ROUGE-2 | PINC | Combined |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash Lite | 4.25 | 29.20 | 11.17 | 74.62 | 38.10 |
| Baidu Qianfan OCR FastFree | 6.14 | 37.65 | 17.14 | 73.76 | 38.01 |
| Liquid LFM 2.5-1.2B Thinking | 1.14 | 10.57 | 2.40 | 89.08 | 44.79 |
| Nvidia Nemotron Nano 9B v2 | 5.47 | 34.52 | 14.11 | 72.94 | 37.47 |
| OpenAI GPT-OSS 20B | 6.64 | 38.88 | 17.20 | 67.02 | 34.62 |

Liquid showed the highest novelty (PINC) but the weakest meaning
preservation. GPT-OSS preserved source structure best (highest BLEU/ROUGE)
but was the least creative (lowest PINC). Automatic metrics alone were not
considered sufficient, motivating the LLM-judge evaluation below.

## Prompt sensitivity (LLM-judge scores, 1-3 scale, out of 265 tweets/model/prompt)

| Prompt | Model | Average | Median | #1 | #2 | #3 |
|---|---|---:|---:|---:|---:|---:|
| 1 | gemini | 1.672 | 1 | 164 | 24 | 77 |
| 1 | liquid | 1.570 | 1 | 164 | 51 | 50 |
| 1 | nvidia | 2.079 | 3 | 117 | 10 | 138 |
| 2 | gemini | 2.230 | 3 | 95 | 14 | 156 |
| 2 | liquid | 1.351 | 1 | 202 | 33 | 30 |
| 2 | nvidia | 2.185 | 3 | 98 | 20 | 147 |
| 3 | gemini | 1.475 | 1 | 196 | 12 | 57 |
| 3 | liquid | 1.106 | 1 | 245 | 12 | 8 |
| 3 | nvidia | 1.660 | 1 | 173 | 9 | 83 |

Prompt 2 was the most effective for human annotators; Prompt 3 was
consistently the worst. Nvidia led human-rated quality (~2.25 average with
Prompt 4), Gemini a close second, Liquid a distant last (~1.3).

Reproduce: `python -m src.postprocessing.significance_tests` confirms both
prompt and model choice have a statistically decisive effect on quality
(Kruskal-Wallis, prompt: statistic=156.699, p=9.4496e-34; model:
statistic=328.303, p=5.1281e-72).

## Per-model rewriting strategies

- **Gemini** -- "explanatory expansion": tends to lengthen the sentence,
  spelling out the sarcasm explicitly.
- **Nvidia** -- "stable precision": keeps sentence length close to the
  source, most consistent.
- **Liquid** -- "unstable reduction": drastically shortens sentences with
  high variance, and loses nearly all lexical overlap with the source
  (near-zero word overlap, ~85% PINC) -- which explains its low human
  quality scores.

Reproduce: `python -m src.postprocessing.linguistic_analysis`

High-quality translations (human score 3.0) tend to keep moderate-to-high
word overlap with the source (~30% median); very aggressive rewriting
correlates with lower quality. Structural features overall correlate only
weakly with quality score (Spearman, mostly r < 0.5) -- sarcasm
interpretation quality is not explainable by simple structural rules.
Reproduce: `python -m src.postprocessing.correlation_heatmap`

## Alt-Test: can the LLM judge replace human annotators?

- **Winning Rate: 0.67**, **Advantage Probability: 0.77** at epsilon=0.2 --
  **PASSED**. A lower epsilon=0.15 gives a self-contradictory result
  (Winning Rate drops to 0.33 while Advantage Probability stays at 0.77),
  which is why epsilon=0.2 was used instead (see `alt_test_reference.md`).
- 3 instances were dropped for having fewer than 2 human annotators.

Reproduce exactly: `python -m src.postprocessing.run_alt_test`

## Human vs. LLM-judge agreement

- **Fleiss' Kappa among the 3 human annotators: 0.282** ("fair agreement")
  -- sarcasm-translation quality is genuinely subjective, even for human
  experts. This motivates the Alt-Test's epsilon tolerance above.
- **Agreement rate (rounded human score == ChatGPT score) by model:**
  Liquid 67.1%, Gemini 50.0%, Nvidia 31.4%. It is easier for humans and the
  LLM judge to agree on an outright failure (Liquid) than on nuanced,
  high-quality output (Nvidia) -- as quality goes up, the automated judge's
  reliability goes down.
- **Confusion matrix** (rounded human score vs. ChatGPT score): ChatGPT
  correctly identifies 61/77 score-1 instances, but misclassifies 47
  instances that humans scored 2 (partially correct) as a complete failure
  (score 1), and almost never uses the middle score itself (only 18 of its
  predictions are a 2). ChatGPT's score matches the rounded human score in
  only 104/210 (49.5%) of cases overall.
- **Semantic rigidity**: high-quality, creative paraphrases from
  Gemini/Nvidia were sometimes scored 1.0 by ChatGPT but 3.0 by humans,
  because they didn't match a rigid expected template.
- **Fluency bias**: Liquid's fluent-sounding but semantically wrong
  translations (e.g. "college is so fun" translated as sincerely positive)
  sometimes fooled ChatGPT into a 3.0 while humans scored them 1.0.

Reproduce: `python -m src.postprocessing.human_llm_agreement` and
`python -m src.postprocessing.extract_case_studies` (saves the concrete
agreement/discrepancy examples to `data/summaries/`).

## Qualitative case studies

- **Safety-filter failure**: Gemini refused to translate a tweet containing
  mild profanity ("bitch") outright; Nvidia successfully rephrased the same
  tweet politely and was rated well by humans. Commercial safety alignment
  can block legitimate NLP work on informal/profane text.
- **Political-correctness bias**: given a sarcastic tweet about racism, both
  Gemini and Nvidia "lectured" the reader instead of neutrally translating
  the sarcasm, and were penalized by human scorers.
- **World-knowledge bottleneck**: a tweet referencing a real-world event
  (Muhammad Ali's death) was misread literally by all 3 models, since none
  had the cultural context; human annotators themselves split on how to
  score it.

Full detail on all of these (with exact tweet text) is in
`project_history.md`, 2026-07-16 meeting section.
