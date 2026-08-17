"""LLM-based sarcasm classification: zero-shot / few-shot / structured reasoning.

Configurable provider/model/temperature/prompt version/few-shot strategy.
Every call is retried on failure or malformed output, optionally cached to
disk (keyed by the exact request content, so a re-run after a transient
failure doesn't re-spend API budget), and runs with bounded thread
concurrency.

Used with `provider="local_hf"` for M2/M3/M4's real DEV/TEST runs (see
`EXPERIMENT_LOG.md`); `provider="openrouter"` remains available as an
optional secondary comparison. Unit-tested with a mocked client
(`tests/test_classification_llm_mocked.py`), matching this repository's
existing convention for API-backed code (see `tests/test_evaluate_with_llm_mocked.py`).

Usage:
    python -m src.classification.llm.run_llm_classification \\
        --experiment-id EXP-002 --mode zero_shot --eval-split dev \\
        --model Qwen/Qwen3-4B-Instruct-2507 --provider local_hf --limit 20   # smoke test first
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config.classification_settings import classification_settings
from src.classification.evaluation.io import save_experiment_artifacts
from src.classification.evaluation.metrics import compute_metrics
from src.classification.llm.client import get_llm_client
from src.classification.llm.few_shot_selection import select_curated_few_shot, select_random_few_shot
from src.classification.llm.schema import parse_label_response
from src.common.prompt_loader import load_prompt

MODES = ("zero_shot", "few_shot", "reasoning")
APPROACH_IDS = {"zero_shot": "M2_llm_zero_shot", "few_shot": "M3_llm_few_shot", "reasoning": "M4_llm_reasoning"}
PROMPT_FILES = {
    "zero_shot": "classification/zero_shot_v1.txt",
    "few_shot": "classification/few_shot_v1.txt",
    "reasoning": "classification/reasoning_v1.txt",
}

CACHE_DIR = classification_settings.project_root / "data" / "llm_cache"


def _cache_key(**kwargs) -> str:
    blob = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_cache(key: str, value: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def format_demonstrations(demo_df: pd.DataFrame) -> str:
    blocks = [
        f'Sentence: "{row["text"]}"\nLabel: {{"label": "{row["label"]}"}}'
        for _, row in demo_df.iterrows()
    ]
    return "\n\n".join(blocks)


def build_messages(mode: str, sentence: str, demonstrations: str | None = None) -> list[dict]:
    system_prompt = load_prompt(PROMPT_FILES[mode])
    if mode == "few_shot":
        system_prompt = system_prompt.replace("{demonstrations}", demonstrations or "")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Sentence: "{sentence}"'},
    ]


def classify_one(
    client,
    model: str,
    mode: str,
    sentence: str,
    temperature: float = 0.0,
    demonstrations: str | None = None,
    max_retries: int = 4,
    wait_seconds: float = 5.0,
    use_cache: bool = True,
) -> dict:
    """One structured-output classification call, with retry + optional
    disk cache. Raises after `max_retries` failed attempts -- callers see
    a hard failure rather than a silently-guessed label."""
    messages = build_messages(mode, sentence, demonstrations)
    key = _cache_key(model=model, mode=mode, temperature=temperature, messages=messages)

    if use_cache:
        cached = _load_cache(key)
        if cached is not None:
            return cached

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model, temperature=temperature, messages=messages
            )
            raw_text = response.choices[0].message.content or ""
            parsed = parse_label_response(raw_text)
            result = {"label": parsed["label"], "raw_response": raw_text}
            usage = getattr(response, "usage", None)
            if usage is not None:
                result["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
                result["completion_tokens"] = getattr(usage, "completion_tokens", None)
            if use_cache:
                _save_cache(key, result)
            return result
        except Exception as exc:  # broad on purpose: retries network, parsing, and API errors alike
            last_error = exc
            if attempt < max_retries:
                time.sleep(wait_seconds)
    raise RuntimeError(f"classify_one failed after {max_retries} attempts: {last_error}")


def run_llm_experiment(
    experiment_id: str,
    mode: str,
    eval_split: str,
    model: str,
    temperature: float = 0.0,
    few_shot_variant: str = "random",
    n_shots: int = 8,
    limit: int | None = None,
    concurrency: int = 4,
    seed: int | None = None,
    use_cache: bool = True,
    splits_dir: Path | None = None,
    results_dir: Path | None = None,
    provider: str = "openrouter",
) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if eval_split == "test":
        print(
            "WARNING: evaluating on TEST. Only do this for a final, frozen "
            "configuration -- never for prompt/few-shot iteration."
        )
    seed = classification_settings.random_seed if seed is None else seed

    splits_dir = splits_dir or classification_settings.splits_dir
    train_df = pd.read_csv(splits_dir / "train.csv")
    eval_df = pd.read_csv(splits_dir / f"{eval_split}.csv")
    if limit:
        eval_df = eval_df.head(limit)

    demonstrations = None
    demo_example_ids: list[str] = []
    if mode == "few_shot":
        selector = select_random_few_shot if few_shot_variant == "random" else select_curated_few_shot
        demo_df = selector(train_df, n_shots, seed)
        demo_example_ids = demo_df["example_id"].tolist()
        demonstrations = format_demonstrations(demo_df)

    client = get_llm_client(provider)
    if provider == "local_hf" and concurrency > 1:
        print("local_hf provider: forcing concurrency=1 (one shared GPU model, not thread-safe to fan out).")
        concurrency = 1

    def _run_row(row: dict) -> dict:
        result = classify_one(
            client,
            model,
            mode,
            row["text"],
            temperature=temperature,
            demonstrations=demonstrations,
            use_cache=use_cache,
        )
        return {
            "example_id": row["example_id"],
            "gold_label": row["label"],
            "predicted_label": result["label"],
        }

    rows = eval_df.to_dict("records")
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_run_row, row) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc=experiment_id):
            results.append(future.result())

    predictions = pd.DataFrame(results).sort_values("example_id").reset_index(drop=True)

    config = {
        "experiment_id": experiment_id,
        "approach": APPROACH_IDS[mode],
        "mode": mode,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "prompt_file": PROMPT_FILES[mode],
        "eval_split": eval_split,
        "n_eval": int(len(eval_df)),
        "seed": seed,
    }
    if mode == "few_shot":
        config.update(
            {
                "few_shot_variant": few_shot_variant,
                "n_shots": n_shots,
                "demo_example_ids": demo_example_ids,
            }
        )

    out_dir = save_experiment_artifacts(experiment_id, config, predictions, results_dir=results_dir)
    metrics = compute_metrics(predictions)
    return {"config": config, "metrics": metrics, "out_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--eval-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=["openrouter", "local_hf"], default="openrouter")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--few-shot-variant", choices=["random", "curated"], default="random")
    parser.add_argument("--n-shots", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test: classify only this many eval rows")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    result = run_llm_experiment(
        args.experiment_id,
        args.mode,
        args.eval_split,
        args.model,
        args.temperature,
        args.few_shot_variant,
        args.n_shots,
        args.limit,
        args.concurrency,
        args.seed,
        use_cache=not args.no_cache,
        provider=args.provider,
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"\nArtifacts written to {result['out_dir']}")


if __name__ == "__main__":
    main()
