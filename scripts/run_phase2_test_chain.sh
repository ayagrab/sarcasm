#!/usr/bin/env bash
# Phase 2: run every frozen non-M1 config once on the sealed TEST split,
# in sequence. M2/M3/M4/M5 all load the same local Qwen model via
# device_map="auto" spanning both Tesla M60s -- see local_client.py's
# max_memory comment -- so they cannot safely run in parallel on this VM
# (would OOM or silently CPU-offload and crawl). M6 is last and is a quick
# eval-only pass (no retraining) against the existing EXP-009 checkpoint.
#
# Resume-aware: skips any step whose results/<experiment_id>/metrics.json
# already exists, so re-running this script after a partial run (e.g. the
# user needed to shut the VM down mid-chain) only does the remaining work.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /mnt/vmadmin/sarcasm-env/venv/bin/activate
export HF_HOME=/mnt/vmadmin/huggingface
export CUDA_VISIBLE_DEVICES=0,1

run_step() {
  local label="$1" exp_id="$2"
  shift 2
  if [ -f "results/${exp_id}/metrics.json" ]; then
    echo "=== [$(date)] ${label} -- SKIPPED, results/${exp_id}/metrics.json already exists ==="
    return 0
  fi
  echo "=== [$(date)] ${label} ==="
  "$@"
}

run_step "M2-TEST (Qwen zero-shot)" EXP-002-TEST \
  python -m src.classification.run_experiment --config configs/EXP-002-TEST.json

run_step "M3-TEST (Qwen few-shot random)" EXP-003-TEST \
  python -m src.classification.run_experiment --config configs/EXP-003-TEST.json

run_step "M4-TEST (Qwen reasoning)" EXP-005-TEST \
  python -m src.classification.run_experiment --config configs/EXP-005-TEST.json

run_step "M5-TEST (DSPy MIPROv2, full recompile)" EXP-008-TEST \
  python -m src.classification.run_experiment --config configs/EXP-008-TEST.json

run_step "M6-TEST (DeBERTa, eval-only against the frozen EXP-009 checkpoint -- no retraining)" EXP-009-TEST \
  env CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_frozen_checkpoint \
  --checkpoint-dir models/EXP-009/best_checkpoint \
  --split test --experiment-id EXP-009-TEST --source-experiment-id EXP-009

echo "=== [$(date)] PHASE2_TEST_CHAIN_COMPLETE ==="
