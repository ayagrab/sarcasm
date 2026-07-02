"""Evaluate model interpretations using an LLM judge with scores 1, 2, or 3."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from config.settings import settings
from src.common.file_utils import read_csv_flexible, save_csv
from src.common.json_utils import extract_json_array
from src.common.openrouter_client import get_openrouter_client
from src.common.prompt_loader import load_prompt

CLASSIFICATION_COLUMN = "classification"


def build_batch_examples(batch_df: pd.DataFrame) -> str:
    """Format a batch of examples for the judge prompt."""
    chunks = []
    for example_number, (_, row) in enumerate(batch_df.iterrows(), start=1):
        chunks.append(f"""
Example {example_number}

Sarcastic sentence:
{row['sarcastic_sentence']}

Model interpretation:
{row['model_interpretation']}

---
""")
    return "\n".join(chunks)


def classify_batch(batch_df: pd.DataFrame, model_id: str, max_retries: int = 5, wait_seconds: int = 15) -> list[int]:
    """Classify a batch using the LLM judge and return one score per row."""
    client = get_openrouter_client()
    judge_prompt = load_prompt("llm_judge_prompt.txt")
    examples = build_batch_examples(batch_df)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                temperature=0,
                messages=[{"role": "system", "content": judge_prompt}, {"role": "user", "content": examples}],
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(extract_json_array(content))
            scores = [int(item["score"]) for item in parsed]
            if len(scores) != len(batch_df):
                raise ValueError(f"Expected {len(batch_df)} scores, got {len(scores)}")
            if any(score not in {1, 2, 3} for score in scores):
                raise ValueError(f"Invalid scores returned: {scores}")
            return scores
        except Exception as exc:
            print(f"Attempt {attempt}/{max_retries} failed: {exc}")
            if attempt == max_retries:
                raise
            time.sleep(wait_seconds)
    raise RuntimeError("Unexpected evaluation failure")


def evaluate_file(input_path: Path, output_path: Path | None = None, model_id: str | None = None, batch_size: int = 10) -> None:
    """Evaluate all unclassified rows in a model-output CSV."""
    df = read_csv_flexible(input_path, ["sarcastic_sentence", "model_interpretation"])
    if CLASSIFICATION_COLUMN not in df.columns:
        df[CLASSIFICATION_COLUMN] = ""
    output_path = output_path or input_path
    model_id = model_id or settings.default_judge_model
    unclassified_indices = df[df[CLASSIFICATION_COLUMN].isna() | (df[CLASSIFICATION_COLUMN].astype(str).str.strip() == "")].index.tolist()

    for start in tqdm(range(0, len(unclassified_indices), batch_size), desc=input_path.name):
        batch_indices = unclassified_indices[start : start + batch_size]
        scores = classify_batch(df.loc[batch_indices], model_id=model_id)
        df.loc[batch_indices, CLASSIFICATION_COLUMN] = scores
        save_csv(df, output_path)


def evaluate_directory(outputs_dir: Path, model_id: str | None = None, batch_size: int = 10) -> None:
    """Evaluate every CSV in a directory tree under data/model_outputs."""
    for csv_file in sorted(outputs_dir.glob("**/*.csv")):
        print(f"Evaluating {csv_file}")
        evaluate_file(csv_file, model_id=model_id, batch_size=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model outputs with an LLM judge.")
    parser.add_argument("--input", type=Path, default=None, help="Single CSV file to evaluate.")
    parser.add_argument("--directory", type=Path, default=settings.model_outputs_dir, help="Directory of CSV files to evaluate when --input is not supplied.")
    parser.add_argument("--model", default=settings.default_judge_model)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    if args.input:
        evaluate_file(args.input, model_id=args.model, batch_size=args.batch_size)
    else:
        evaluate_directory(args.directory, model_id=args.model, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
