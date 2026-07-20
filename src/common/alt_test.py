"""The Alt-Test: statistically justifying replacing human annotators with an LLM.

This is the reference implementation from:
Calderon, Reichart & Dror (2025), "The Alternative Annotator Test for
LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with
LLMs" (ACL 2025). https://arxiv.org/abs/2501.10970

Used in this project to check whether the LLM judge (see
`src/evaluation/evaluate_with_llm.py`) can replace the three human annotators
for scoring sarcasm interpretations -- see `docs/alt_test_reference.md` and
`docs/project_history.md` (2026-07-09 / 2026-07-16 meetings) for the
resulting Winning Rate / Advantage Probability and how epsilon was chosen.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Union

import numpy as np
from scipy.stats import ttest_1samp


def accuracy(pred: Any, annotations: List[Any]) -> float:
    """Fraction of annotations that exactly match the prediction."""
    return float(np.mean([pred == ann for ann in annotations]))


def neg_rmse(pred: Union[int, float], annotations: List[Union[int, float]]) -> float:
    """Negative RMSE between a numeric prediction and a list of numeric annotations."""
    return -1 * float(np.sqrt(np.mean([(pred - ann) ** 2 for ann in annotations])))


def sim(pred: str, annotations: List[str], similarity_func: Callable) -> float:
    """Average of a custom similarity function between the prediction and each annotation."""
    return float(np.mean([similarity_func(pred, ann) for ann in annotations]))


def _ttest(indicators: List[float], epsilon: float) -> float:
    return ttest_1samp(indicators, epsilon, alternative="less").pvalue


def _by_procedure(p_values: List[float], q: float) -> List[int]:
    """Benjamini-Yekutieli procedure: indices of p-values rejected at FDR level q."""
    p_values = np.array(p_values, dtype=float)
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    harmonic_sum = np.sum(1.0 / np.arange(1, m + 1))
    thresholds = (np.arange(1, m + 1) / m) * (q / harmonic_sum)
    max_i = -1
    for i in range(m):
        if sorted_pvals[i] <= thresholds[i]:
            max_i = i
    if max_i == -1:
        return []
    return list(sorted_indices[: max_i + 1])


def alt_test(
    llm_annotations: Dict[Union[int, str], Any],
    humans_annotations: Dict[Union[int, str], Dict[Union[int, str], Any]],
    scoring_function: Union[str, Callable] = "accuracy",
    epsilon: float = 0.2,
    q_fdr: float = 0.05,
    min_humans_per_instance: int = 2,
    min_instances_per_human: int = 30,
) -> tuple[float, float]:
    """Run the alt-test: can the LLM replace the human annotators?

    Leave-one-human-out: for each annotator, compare how well the LLM vs. the
    excluded human aligns with the remaining annotators on the instances they
    both labeled.

    Returns
    -------
    winning_rate : float
        Fraction of human annotators the LLM "wins" against (aligns better
        with the remaining annotators). The LLM passes if `winning_rate >= 0.5`.
    advantage_prob : float
        Estimated probability the LLM is at least as good as a randomly
        chosen human annotator. Use this to compare LLMs (higher is better).
    """
    if isinstance(scoring_function, str):
        if scoring_function == "accuracy":
            scoring_function = accuracy
        elif scoring_function == "neg_rmse":
            scoring_function = neg_rmse
        else:
            raise ValueError("Unknown scoring function")

    instances_by_human: Dict[Any, List[Any]] = {}
    humans_by_instance: Dict[Any, List[Any]] = {}
    for human, annotations in humans_annotations.items():
        instances_by_human[human] = list(annotations.keys())
        for instance, _ in annotations.items():
            humans_by_instance.setdefault(instance, []).append(human)

    instances_to_keep = {
        instance
        for instance in humans_by_instance
        if len(humans_by_instance[instance]) >= min_humans_per_instance and instance in llm_annotations
    }
    if len(instances_to_keep) < len(humans_by_instance):
        dropped = len(humans_by_instance) - len(instances_to_keep)
        print(f"Dropped {dropped} instances with less than {min_humans_per_instance} annotators.")
    instances_by_human = {h: [i for i in insts if i in instances_to_keep] for h, insts in instances_by_human.items()}
    humans_by_instance = {i: hs for i, hs in humans_by_instance.items() if i in instances_to_keep}

    p_values, advantage_probs, evaluated_humans = [], [], []
    for excluded_human in humans_annotations:
        instances = [i for i in instances_by_human[excluded_human] if i in llm_annotations]
        if len(instances) < min_instances_per_human:
            print(f"Skipping annotator {excluded_human} with only {len(instances)} instances < {min_instances_per_human}.")
            continue

        llm_indicators, excluded_indicators = [], []
        for instance in instances:
            human_ann = humans_annotations[excluded_human][instance]
            llm_ann = llm_annotations[instance]
            remaining_anns = [humans_annotations[h][instance] for h in humans_by_instance[instance] if h != excluded_human]
            human_score = scoring_function(human_ann, remaining_anns)
            llm_score = scoring_function(llm_ann, remaining_anns)
            llm_indicators.append(1 if llm_score >= human_score else 0)
            excluded_indicators.append(1 if human_score >= llm_score else 0)

        diff_indicators = [exc - llm for exc, llm in zip(excluded_indicators, llm_indicators)]
        p_values.append(_ttest(diff_indicators, epsilon))
        advantage_probs.append(float(np.mean(llm_indicators)))
        evaluated_humans.append(excluded_human)

    rejected_indices = _by_procedure(p_values, q_fdr)
    advantage_prob = float(np.mean(advantage_probs))
    winning_rate = len(rejected_indices) / len(evaluated_humans)
    return winning_rate, advantage_prob
