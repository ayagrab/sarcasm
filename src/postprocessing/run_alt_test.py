"""Run the alt-test to check whether the LLM judge can replace human annotators.

See `src/common/alt_test.py` for the algorithm and `docs/alt_test_reference.md`
for how epsilon was chosen and how to interpret the results.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from config.settings import settings
from src.common.alt_test import alt_test


def run_alt_test(
    humans_path: Path,
    llm_path: Path,
    scoring_function: str = "neg_rmse",
    epsilon: float = 0.2,
    min_instances_per_human: int = 50,
) -> tuple[float, float]:
    """Load the two annotation files and return (winning_rate, advantage_prob)."""
    humans_data = json.loads(humans_path.read_text(encoding="utf-8"))
    llm_data = json.loads(llm_path.read_text(encoding="utf-8"))
    return alt_test(
        llm_annotations=llm_data,
        humans_annotations=humans_data,
        scoring_function=scoring_function,
        epsilon=epsilon,
        min_instances_per_human=min_instances_per_human,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the alt-test on the human vs. LLM judge annotations.")
    parser.add_argument("--humans", type=Path, default=settings.data_dir / "alt_test" / "humans_annotations.json")
    parser.add_argument("--llm", type=Path, default=settings.data_dir / "alt_test" / "llm_annotations.json")
    parser.add_argument("--scoring-function", default="neg_rmse", choices=["accuracy", "neg_rmse"])
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--min-instances-per-human", type=int, default=50)
    args = parser.parse_args()

    winning_rate, advantage_prob = run_alt_test(
        args.humans, args.llm, args.scoring_function, args.epsilon, args.min_instances_per_human
    )
    print("-" * 40)
    print("Alt-Test Results:")
    print(f"Winning Rate: {winning_rate:.2f}")
    print(f"Advantage Probability: {advantage_prob:.2f}")
    print("-" * 40)
    if winning_rate >= 0.5:
        print("PASSED - the LLM can replace the human annotators for this task.")
    else:
        print("FAILED - the LLM is not yet good enough to replace the human annotators.")


if __name__ == "__main__":
    main()
