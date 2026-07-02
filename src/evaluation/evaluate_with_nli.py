"""Evaluate interpretations with an NLI model.

This is an alternative automatic evaluation method. It treats the sarcastic
sentence as the premise and the interpretation as the hypothesis, then checks
whether entailment is stronger than contradiction.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config.settings import settings
from src.common.file_utils import read_csv_flexible, save_csv
from src.common.prompt_loader import load_prompt


def evaluate_with_nli(input_path: Path, output_path: Path, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli") -> None:
    """Add an `nli_success` column: 1 if entailment > contradiction, else 0."""
    df = read_csv_flexible(input_path, ["sarcastic_sentence", "model_interpretation"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    premise_template = load_prompt("nli_premise_template.txt")
    hypothesis_template = load_prompt("nli_hypothesis_template.txt")
    results = []

    for _, row in df.iterrows():
        interpretation = row.get("model_interpretation", "")
        if pd.isna(row["sarcastic_sentence"]) or pd.isna(interpretation) or str(interpretation).strip().lower() == "<unset>":
            results.append(0)
            continue
        premise = premise_template.format(sarcastic_sentence=row["sarcastic_sentence"])
        hypothesis = hypothesis_template.format(model_interpretation=interpretation)
        inputs = tokenizer(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
        label_probs = {model.config.id2label[i].lower(): probabilities[i].item() for i in range(len(probabilities))}
        results.append(1 if label_probs.get("entailment", 0.0) > label_probs.get("contradiction", 0.0) else 0)

    df["nli_success"] = results
    save_csv(df, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate output CSV with an NLI model.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate_with_nli(args.input, args.output)

if __name__ == "__main__":
    main()
