"""Single entry point for launching any classification experiment.

    python -m src.classification.run_experiment --config configs/<name>.json [--experiment-id EXP-0NN]

Reads a JSON config (see `configs/`), dispatches to the right approach
module based on the config's `approach_family` field, and runs it. Every
config file is a complete, versioned record of one experiment's settings
-- no model/prompt/optimizer/hyperparameter choice is hardcoded here, so
switching configuration never requires touching code (task Section 12).
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

DISPATCH = {
    "tfidf": "src.classification.classical.tfidf_baseline:run_tfidf_experiment",
    "llm": "src.classification.llm.run_llm_classification:run_llm_experiment",
    "dspy": "src.classification.dspy_pipeline.run_dspy:run_dspy_experiment",
    "transformer": "src.classification.transformer.finetune:finetune_and_evaluate",
}


def _import_callable(path: str):
    module_path, func_name = path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def run_from_config(config_path: Path, experiment_id: str | None = None) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config = {k: v for k, v in config.items() if not k.startswith("_")}

    approach_family = config.pop("approach_family")
    experiment_id = experiment_id or config.pop("experiment_id")
    if approach_family not in DISPATCH:
        raise ValueError(f"Unknown approach_family {approach_family!r} (expected one of {list(DISPATCH)})")
    func = _import_callable(DISPATCH[approach_family])

    if approach_family == "transformer":
        from src.classification.transformer.finetune import TransformerConfig

        eval_split = config.pop("eval_split", "dev")
        transformer_config = TransformerConfig(experiment_id=experiment_id, **config)
        return func(transformer_config, eval_split=eval_split)

    return func(experiment_id=experiment_id, **config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", default=None, help="Overrides the config file's experiment_id")
    args = parser.parse_args()

    result = run_from_config(args.config, args.experiment_id)
    print(json.dumps(result["metrics"], indent=2))
    print(f"\nArtifacts written to {result['out_dir']}")


if __name__ == "__main__":
    main()
