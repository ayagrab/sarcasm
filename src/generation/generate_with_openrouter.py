"""Generate non-sarcastic interpretations using OpenRouter models."""
from __future__ import annotations
import argparse, time
from pathlib import Path
from config.settings import settings
from src.common.file_utils import read_csv_flexible, save_csv
from src.common.openrouter_client import get_openrouter_client
from src.common.prompt_loader import load_prompt


def generate_interpretations(input_path: Path, output_path: Path, model_id: str, start_row: int = 0, end_row: int | None = None, sleep_seconds: float = 1.0) -> None:
    """Generate interpretations with an OpenRouter model and save progress continuously."""
    df = read_csv_flexible(input_path, ["sarcastic_sentence", "model_interpretation"])
    if "model_interpretation" not in df.columns:
        df["model_interpretation"] = ""
    client = get_openrouter_client()
    prompt_template = load_prompt("interpret_sarcasm_prompt.txt")
    end = len(df) if end_row is None else min(end_row, len(df))

    for row_index in range(start_row, end):
        sentence = str(df.loc[row_index, "sarcastic_sentence"]).strip()
        existing = str(df.loc[row_index, "model_interpretation"]).strip()
        if existing and existing not in {"ERROR", "AI_PROCESSING_ERROR", "nan"}:
            continue
        prompt = prompt_template.format(sarcastic_sentence=sentence)
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You convert sarcastic tweets into sincere interpretations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            df.loc[row_index, "model_interpretation"] = response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"Error on row {row_index}: {exc}")
            df.loc[row_index, "model_interpretation"] = "ERROR"
        save_csv(df, output_path)
        time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sarcasm interpretations with OpenRouter.")
    parser.add_argument("--input", type=Path, default=settings.processed_data_dir / "clean_sarcastic_sentences.csv")
    parser.add_argument("--output", type=Path, default=settings.model_outputs_dir / "experiment_new" / "openrouter.csv")
    parser.add_argument("--model", default=settings.default_openrouter_model)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()
    generate_interpretations(args.input, args.output, args.model, args.start_row, args.end_row, args.sleep)

if __name__ == "__main__":
    main()
