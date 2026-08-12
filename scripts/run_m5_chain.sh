#!/usr/bin/env bash
# Run M5 (DSPy + local Qwen) back-to-back on the VM, once the M3/M4 chain
# (scripts/run_m3_m4_chain.sh) has finished and the GPU is free: adapter
# smoke test, EXP-006 (dspy.Predict, unoptimized baseline), EXP-007
# (BootstrapFewShot), EXP-008 (MIPROv2). EXP-007/008 already use the
# small/conservative budgets fixed in their config files
# (max_bootstrapped_demos=4 / auto="light"), per the "start conservative,
# expand only if justified" instruction -- whether to expand beyond that
# is a judgment call made *after* inspecting these three results, not
# automated here. The smoke test gates the rest: if the adapter is broken,
# `set -e` stops the chain before any full-DEV (1,340-call) run wastes
# hours of compute on it.
#
# Run this ON THE VM, inside the activated sarcasm-env, from the repo root:
#   nohup bash scripts/run_m5_chain.sh > logs/m5_chain.log 2>&1 &
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

bash scripts/verify_kernel.sh

echo "=== $(date -u +%FT%TZ) starting M5-DSPy-smoke-test ==="
python scripts/smoke_test_dspy.py 2>&1 | tee logs/M5-dspy-smoke-test.log
echo "=== $(date -u +%FT%TZ) finished M5-DSPy-smoke-test ==="

run_step() {
  local config="$1"
  local label="$2"
  echo "=== $(date -u +%FT%TZ) starting $label ($config) ==="
  python -m src.classification.run_experiment --config "$config" 2>&1 | tee "logs/${label}-dev.log"
  echo "=== $(date -u +%FT%TZ) finished $label ==="
}

run_step configs/dspy_predict.json EXP-006-dspy-predict
run_step configs/dspy_bootstrap_few_shot.json EXP-007-dspy-bootstrap
run_step configs/dspy_mipro_v2.json EXP-008-dspy-mipro

echo "Chain complete: EXP-006, EXP-007, EXP-008 (M5/DSPy) all finished."
