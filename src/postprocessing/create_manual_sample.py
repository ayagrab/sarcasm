"""Create a random manual-scoring sample across model outputs."""
from __future__ import annotations
import argparse, random
from pathlib import Path
import pandas as pd
from config.settings import settings
from src.common.file_utils import save_csv

MODEL_KEYWORDS = {"gemini": "gemini", "nvidia": "nvidia", "liquid": "liquid"}


def find_one_csv(folder: Path, keyword: str) -> Path:
    """Find exactly one CSV file in a folder containing keyword in its name."""
    matches = list(folder.glob(f"*{keyword}*.csv"))
    if not matches:
        raise FileNotFoundError(f"No CSV file containing '{keyword}' found in {folder}")
    if len(matches) > 1:
        raise ValueError(f"More than one CSV file containing '{keyword}' found in {folder}: {matches}")
    return matches[0]


def create_sample(outputs_dir: Path, output_file: Path, sample_size: int = 70, seed: int | None = 42) -> pd.DataFrame:
    """Build a manual scoring CSV with outputs from Gemini, Nvidia and Liquid."""
    if seed is not None:
        random.seed(seed)
    experiments = sorted([p for p in outputs_dir.iterdir() if p.is_dir()])
    if not experiments:
        raise FileNotFoundError(f"No experiment folders found under {outputs_dir}")
    first_gemini = pd.read_csv(find_one_csv(experiments[0], MODEL_KEYWORDS["gemini"]), encoding="utf-8-sig")
    random_indices = random.sample(range(len(first_gemini)), min(sample_size, len(first_gemini)))
    rows = []
    for idx in random_indices:
        experiment = random.choice(experiments)
        gemini_df = pd.read_csv(find_one_csv(experiment, MODEL_KEYWORDS["gemini"]), encoding="utf-8-sig")
        nvidia_df = pd.read_csv(find_one_csv(experiment, MODEL_KEYWORDS["nvidia"]), encoding="utf-8-sig")
        liquid_df = pd.read_csv(find_one_csv(experiment, MODEL_KEYWORDS["liquid"]), encoding="utf-8-sig")
        rows.append({
            "experiment": experiment.name,
            "sarcastic_sentence": gemini_df.loc[idx, "sarcastic_sentence"],
            "gemini_interpretation": gemini_df.loc[idx, "model_interpretation"],
            "gemini_score": "",
            "nvidia_interpretation": nvidia_df.loc[idx, "model_interpretation"],
            "nvidia_score": "",
            "liquid_interpretation": liquid_df.loc[idx, "model_interpretation"],
            "liquid_score": "",
        })
    sample_df = pd.DataFrame(rows)
    save_csv(sample_df, output_file)
    return sample_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a random manual scoring sample.")
    parser.add_argument("--outputs-dir", type=Path, default=settings.model_outputs_dir)
    parser.add_argument("--output", type=Path, default=settings.manual_scoring_dir / "random_70_for_manual_scoring.csv")
    parser.add_argument("--sample-size", type=int, default=70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sample_df = create_sample(args.outputs_dir, args.output, args.sample_size, args.seed)
    print(f"Created {len(sample_df)} rows at {args.output}")

if __name__ == "__main__":
    main()
