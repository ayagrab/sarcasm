"""Phase 4, M2/M3/M4 legs: zero-transfer evaluation of the frozen
Qwen3-4B prompting approaches (zero-shot / few-shot / structured
reasoning -- Part II's EXP-002/EXP-003/EXP-005 winning configs) on SIGN
Test. No retraining, no re-selection of few-shot demonstrations against
SIGN -- reuses `classify_one`/`build_messages`/`format_demonstrations`/
`select_random_few_shot` directly from
`src.classification.llm.run_llm_classification` (imported, not
duplicated) so the exact same prompt templates and (for few-shot) demo
-selection logic run unchanged, just against SIGN Test instead of
Dataset A. The few-shot demonstrations still come from **Dataset A
TRAIN** (same seed=42, n_shots=8, random variant as the frozen EXP-003
config) -- SIGN is never used to pick demonstrations, only as the
held-out eval set.

Requires the Azure VM (CUDA) via `provider="local_hf"`
(`src.classification.llm.local_client.LocalHFClient` hard-requires a GPU).

Run (all three modes in one process, so the ~8GB model loads only once):
    python -m src.sign.zero_transfer.run_llm_zero_transfer
"""
from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from config.classification_settings import classification_settings
from src.classification.llm.few_shot_selection import select_random_few_shot
from src.classification.llm.run_llm_classification import (
    PROMPT_FILES,
    build_messages,
    classify_one,
    format_demonstrations,
)
from src.classification.llm.client import get_llm_client
from src.sign.data.load_sign import load_family_table
from src.sign.zero_transfer.io import save_zero_transfer_result

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
EXPERIMENT_IDS = {"zero_shot": "EXP-SIGN-012", "few_shot": "EXP-SIGN-013", "reasoning": "EXP-SIGN-014"}
FROZEN_SOURCE = {"zero_shot": "EXP-002", "few_shot": "EXP-003", "reasoning": "EXP-005"}


def run_mode(client, mode: str, sign_test: pd.DataFrame, train_df: pd.DataFrame, seed: int, smoke: bool = False) -> None:
    demonstrations = None
    demo_example_ids: list[str] = []
    if mode == "few_shot":
        demo_df = select_random_few_shot(train_df, n_shots=8, seed=seed)
        demo_example_ids = demo_df["example_id"].tolist()
        demonstrations = format_demonstrations(demo_df)

    results = []
    for row in tqdm(sign_test.to_dict("records"), desc=mode):
        result = classify_one(
            client, MODEL, mode, row["text"], temperature=0.0, demonstrations=demonstrations, use_cache=True
        )
        results.append({"example_id": row["example_id"], "gold_label": row["label"], "predicted_label": result["label"]})

    predictions = pd.DataFrame(results)
    experiment_id = EXPERIMENT_IDS[mode] + ("-SMOKE" if smoke else "")
    config = {
        "experiment_id": experiment_id,
        "approach": f"{mode}_zero_transfer",
        "frozen_config_reused": f"configs/*_qwen_local.json ({FROZEN_SOURCE[mode]})",
        "model": MODEL,
        "provider": "local_hf",
        "temperature": 0.0,
        "prompt_file": PROMPT_FILES[mode],
        "train_data": "NONE (Dataset A TRAIN only used for few-shot demo selection, not fitting)",
        "eval_data": "SIGN Test, all roles (data/sign/family_table_test.csv, n=1735)",
        "seed": seed,
    }
    if mode == "few_shot":
        config.update({"few_shot_variant": "random", "n_shots": 8, "demo_example_ids": demo_example_ids})
    save_zero_transfer_result(experiment_id, config, predictions, sign_test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 / M2-M4: zero-transfer Qwen prompting on SIGN Test.")
    parser.add_argument("--modes", nargs="+", default=["zero_shot", "few_shot", "reasoning"], choices=list(EXPERIMENT_IDS))
    parser.add_argument("--seed", type=int, default=classification_settings.random_seed)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test: classify only this many SIGN Test rows")
    args = parser.parse_args()

    train_df = pd.read_csv(classification_settings.splits_dir / "train.csv", encoding="utf-8-sig")
    sign_test = load_family_table("test")
    if args.limit:
        sign_test = sign_test.head(args.limit)

    print("Loading local Qwen3-4B-Instruct-2507 client (one-time load for all modes) ...")
    client = get_llm_client("local_hf")

    for mode in args.modes:
        run_mode(client, mode, sign_test, train_df, args.seed, smoke=bool(args.limit))


if __name__ == "__main__":
    main()
