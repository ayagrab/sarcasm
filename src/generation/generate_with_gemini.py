"""Generate non-sarcastic interpretations using Gemini."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import pandas as pd
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from config.settings import settings
from src.common.file_utils import read_csv_flexible, save_csv
from src.common.gemini_client import get_gemini_model
from src.common.prompt_loader import load_prompt


def generate_interpretations(input_path: Path, output_path: Path, start_row: int = 0, end_row: int | None = None, sleep_seconds: float = 2.0) -> None:
    """Generate interpretations for a slice of rows and save progress after each row."""
    df = read_csv_flexible(input_path, ["sarcastic_sentence", "model_interpretation"])
    if "model_interpretation" not in df.columns:
        df["model_interpretation"] = ""
    prompt_template = load_prompt("interpret_sarcasm_prompt.txt")
    model = get_gemini_model()
    end = len(df) if end_row is None else min(end_row, len(df))

    for row_index in range(start_row, end):
        sentence = str(df.loc[row_index, "sarcastic_sentence"]).strip()
        existing = str(df.loc[row_index, "model_interpretation"]).strip()
        if existing and existing not in {"ERROR", "AI_PROCESSING_ERROR", "nan"}:
            continue
        prompt = prompt_template.format(sarcastic_sentence=sentence)
        try:
            response = model.generate_content(
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
            df.loc[row_index, "model_interpretation"] = response.text.strip() if response.text else "[AI Refused]"
        except Exception as exc:
            print(f"Error on row {row_index}: {exc}")
            df.loc[row_index, "model_interpretation"] = "ERROR"
        save_csv(df, output_path)
        time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sarcasm interpretations with Gemini.")
    parser.add_argument("--input", type=Path, default=settings.processed_data_dir / "clean_sarcastic_sentences.csv")
    parser.add_argument("--output", type=Path, default=settings.model_outputs_dir / "experiment_new" / "gemini.csv")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()
    generate_interpretations(args.input, args.output, args.start_row, args.end_row, args.sleep)

if __name__ == "__main__":
    main()
