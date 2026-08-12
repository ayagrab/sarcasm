"""DSPy-based sarcasm classification: Predict / BootstrapFewShot / MIPROv2.

Default provider is `local_hf` -- the same local Qwen3-4B-Instruct-2507
model used by the manual-prompt LLM experiments (`src.classification.llm`),
via `src.classification.dspy_pipeline.local_lm.LocalQwenLM` (a
`dspy.BaseLM` subclass, no OpenAI-compatible server, no vLLM -- Tesla M60
doesn't support vLLM's modern backend). `provider="openrouter"` remains
available as an optional secondary comparison (needs `OPENROUTER_API_KEY`).

TRAIN is used for optimization/bootstrapping; DEV is the optimizer's
validation metric (MIPROv2) or otherwise held out; TEST is only touched
for the final, frozen-program evaluation, never during optimization or
candidate selection.

Usage (local Qwen, default):
    python -m src.classification.dspy_pipeline.run_dspy \\
        --experiment-id EXP-006 --optimizer predict --eval-split dev \\
        --model Qwen/Qwen3-4B-Instruct-2507
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config.classification_settings import classification_settings
from src.classification.dspy_pipeline.signatures import HAS_DSPY, build_signature
from src.classification.evaluation.io import save_experiment_artifacts
from src.classification.evaluation.metrics import compute_metrics

OPTIMIZERS = ("predict", "bootstrap_few_shot", "mipro_v2")
APPROACH_ID = "M5_dspy"


def _require_dspy():
    if not HAS_DSPY:
        raise RuntimeError(
            "dspy is not installed. Run `pip install -r requirements-classification.txt` "
            "(see EXPERIMENT_LOG.md, Stage A readiness report)."
        )
    import dspy

    return dspy


def _configure_lm(dspy_module, model: str, temperature: float, provider: str = "local_hf"):
    if provider == "local_hf":
        from src.classification.dspy_pipeline.local_lm import LocalQwenLM

        lm = LocalQwenLM(checkpoint=model, temperature=temperature)
        dspy_module.configure(lm=lm)
        return lm

    if not classification_settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to your .env file "
            "(see EXPERIMENT_LOG.md, BLOCKER-1)."
        )
    lm = dspy_module.LM(
        f"openrouter/{model}",
        api_key=classification_settings.openrouter_api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
    )
    dspy_module.configure(lm=lm)
    return lm


def _to_dspy_examples(df: pd.DataFrame, dspy_module):
    return [
        dspy_module.Example(sentence=row["text"], label=row["label"]).with_inputs("sentence")
        for _, row in df.iterrows()
    ]


def _label_match_metric(example, prediction, trace=None) -> bool:
    return getattr(prediction, "label", None) == example.label


def build_program(
    dspy_module,
    optimizer: str,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    optimizer_config: dict | None = None,
    seed: int = 42,
):
    """NOTE on LM-call budget: this project's TRAIN/DEV splits are large
    (6,706 / 1,340 examples). Passing them to DSPy optimizers uncapped
    would mean thousands of sequential LM calls on hardware that runs at
    ~1-2s/call (Tesla M60, no tensor cores) -- hours of wall-clock time
    per optimizer. Both `BootstrapFewShot` and `MIPROv2` here are
    deliberately capped via `trainset_sample_size` / `valset_sample_size`
    in `optimizer_config` (defaults: 150 / 100), matching the "small
    budget first, expand only if it pays off" instruction. See
    EXPERIMENT_LOG.md, Stage B, M5 for the recorded LM-call estimate and
    actual wall-clock time of each run."""
    signature = build_signature()
    program = dspy_module.Predict(signature)

    if optimizer == "predict":
        return program

    optimizer_config = optimizer_config or {}
    trainset_sample_size = optimizer_config.get("trainset_sample_size", 150)
    train_sample = train_df.sample(n=min(trainset_sample_size, len(train_df)), random_state=seed)
    trainset = _to_dspy_examples(train_sample, dspy_module)

    if optimizer == "bootstrap_few_shot":
        from dspy.teleprompt import BootstrapFewShot

        teleprompter = BootstrapFewShot(
            metric=_label_match_metric,
            max_bootstrapped_demos=optimizer_config.get("max_bootstrapped_demos", 4),
            max_labeled_demos=optimizer_config.get("max_labeled_demos", 8),
            max_rounds=optimizer_config.get("max_rounds", 1),
        )
        return teleprompter.compile(program, trainset=trainset)

    if optimizer == "mipro_v2":
        from dspy.teleprompt import MIPROv2

        valset_sample_size = optimizer_config.get("valset_sample_size", 100)
        dev_sample = dev_df.sample(n=min(valset_sample_size, len(dev_df)), random_state=seed)
        devset = _to_dspy_examples(dev_sample, dspy_module)
        teleprompter = MIPROv2(metric=_label_match_metric, auto=optimizer_config.get("auto", "light"))
        return teleprompter.compile(program, trainset=trainset, valset=devset)

    raise ValueError(f"Unknown optimizer: {optimizer}")


def run_dspy_experiment(
    experiment_id: str,
    optimizer: str,
    eval_split: str,
    model: str,
    temperature: float = 0.0,
    optimizer_config: dict | None = None,
    seed: int | None = None,
    provider: str = "local_hf",
) -> dict:
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    if eval_split == "test":
        print(
            "WARNING: evaluating on TEST. Only do this for a final, frozen "
            "DSPy program -- never during/after repeated optimization attempts."
        )
    dspy_module = _require_dspy()
    _configure_lm(dspy_module, model, temperature, provider)
    seed = classification_settings.random_seed if seed is None else seed

    # Resolve LM-call-budget defaults up front so the saved config reflects
    # what actually ran, not just what the caller happened to specify.
    optimizer_config = dict(optimizer_config or {})
    if optimizer in ("bootstrap_few_shot", "mipro_v2"):
        optimizer_config.setdefault("trainset_sample_size", 150)
    if optimizer == "mipro_v2":
        optimizer_config.setdefault("valset_sample_size", 100)

    splits_dir = classification_settings.splits_dir
    train_df = pd.read_csv(splits_dir / "train.csv")
    dev_df = pd.read_csv(splits_dir / "dev.csv")
    eval_df = pd.read_csv(splits_dir / f"{eval_split}.csv")

    program = build_program(dspy_module, optimizer, train_df, dev_df, optimizer_config, seed=seed)

    results = []
    for _, row in eval_df.iterrows():
        prediction = program(sentence=row["text"])
        results.append(
            {
                "example_id": row["example_id"],
                "gold_label": row["label"],
                "predicted_label": prediction.label,
            }
        )
    predictions = pd.DataFrame(results)

    config = {
        "experiment_id": experiment_id,
        "approach": APPROACH_ID,
        "optimizer": optimizer,
        "optimizer_config": optimizer_config or {},
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "eval_split": eval_split,
        "n_eval": int(len(eval_df)),
        "seed": seed,
        "dspy_version": getattr(dspy_module, "__version__", "unknown"),
    }
    if optimizer != "predict":
        config["train_size"] = int(len(train_df))
        config["dev_size_used_for_optimization"] = int(len(dev_df))

    out_dir = save_experiment_artifacts(experiment_id, config, predictions)
    if optimizer != "predict":
        try:
            program.save(str(Path(out_dir) / "compiled_program.json"))
        except Exception as exc:  # dspy save-format specifics may vary by version
            print(f"Warning: could not save compiled DSPy program: {exc}")

    metrics = compute_metrics(predictions)
    return {"config": config, "metrics": metrics, "out_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, required=True)
    parser.add_argument("--eval-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=["local_hf", "openrouter"], default="local_hf")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result = run_dspy_experiment(
        args.experiment_id,
        args.optimizer,
        args.eval_split,
        args.model,
        args.temperature,
        seed=args.seed,
        provider=args.provider,
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"\nArtifacts written to {result['out_dir']}")


if __name__ == "__main__":
    main()
