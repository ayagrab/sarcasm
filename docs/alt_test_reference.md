# Alt-Test — Reference

The Alt-Test is a statistical procedure to determine whether an LLM's
annotations can replace human annotators. This project uses it to check
whether the LLM judge (`src/evaluation/evaluate_with_llm.py`) can replace
the three human annotators who scored sarcasm interpretations by hand (see
`docs/meeting_notes_summary.md`, 2026-07-09 and 2026-07-16 meetings, for the
full methodology and results: Winning Rate 0.67, Advantage Probability 0.77).

## Source

Calderon, Reichart & Dror (2025), *"The Alternative Annotator Test for
LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with
LLMs"*, ACL 2025. https://arxiv.org/abs/2501.10970

The algorithm implementation in `src/common/alt_test.py` is adapted directly
from the paper's reference code
(the team's working copy of the paper's repo, `alt_test_example.ipynb`, is
where this project's own Alt-Test analysis was originally run). Please cite
the paper above if this method is used elsewhere.

## How it works (short version)

- Leave-one-annotator-out: for each human annotator, exclude them and check
  whether the LLM or the excluded human better matches the *remaining*
  annotators, using a chosen scoring function (this project uses
  `neg_rmse`, since scores are numeric 1.0/2.0/3.0).
- **Winning Rate**: the fraction of human annotators the LLM "wins" against.
  The LLM passes if `winning_rate >= 0.5`.
- **Advantage Probability (ρ)**: estimated probability the LLM is at least
  as good as a randomly chosen human annotator. Used to compare LLMs against
  each other (higher is better), not as a pass/fail threshold.
- **epsilon (ε)**: a cost-benefit tolerance that credits the LLM for being
  cheaper/faster. Rule of thumb from the paper's FAQ:
  - `0.2` for expert annotators (reliable but expensive)
  - `0.15` for skilled annotators (e.g. trained students)
  - `0.1` for crowd-workers
  - Lower human inter-annotator agreement (IAA) calls for a *smaller* ε.
  This project uses **ε = 0.2**, and found that a lower ε = 0.15 gives a
  self-contradictory result (Winning Rate drops to 0.33 while Advantage
  Probability stays at 0.77) -- see the 2026-07-16 meeting notes for why 0.2
  was chosen instead.

## Where things live in this project

- **`src/common/alt_test.py`** — the `alt_test()` function and its scoring
  helpers (`accuracy`, `neg_rmse`, `sim`).
- **`data/alt_test/humans_annotations.json`** — the 3 team members' (aya,
  anat, yehoraz) scores, keyed by annotator then by instance
  (`"<prompt>_<model>_<sarcastic sentence>"`).
- **`data/alt_test/llm_annotations.json`** — the LLM judge's scores for the
  same instances.
- **`src/postprocessing/run_alt_test.py`** — loads both files and prints the
  Winning Rate / Advantage Probability.

Run:

```bash
python -m src.postprocessing.run_alt_test
```

This reproduces the project's documented result exactly: `Dropped 3
instances with less than 2 annotators`, Winning Rate 0.67, Advantage
Probability 0.77, PASSED.

## What was intentionally left out

The team's working copy of the paper's repo also included the paper's own
generic example datasets (10k_prompts, cebab_aspects, cebab_stars, framing,
kilogram, lesion, lgbteen, mtbench, summeval, wax — ~65MB, some with images)
and a generic simulation tool for choosing sample size/epsilon. These are
useful if you want to explore the method further, but they aren't part of
the sarcasm project's own data or results, so they weren't copied in here.
