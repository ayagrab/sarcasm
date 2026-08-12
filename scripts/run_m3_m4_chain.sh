#!/usr/bin/env bash
# Run the remaining M3/M4 Phase 1 (DEV-only) Stage B experiments back-to-
# back on the VM: EXP-003 (M3 random few-shot), EXP-004 (M3 curated
# few-shot), EXP-005 (M4 structured reasoning). Each step only starts
# after the previous one exits 0, so a failure stops the chain rather than
# silently skipping ahead. Run this ON THE VM, inside the activated
# sarcasm-env, from the repo root, e.g.:
#   nohup bash scripts/run_m3_m4_chain.sh > logs/m3_m4_chain.log 2>&1 &
#
# M5 (DSPy) continues in scripts/run_m5_chain.sh, run separately after this
# one finishes -- deliberately not appended here, so editing this file
# on disk mid-run (this script gets git-tracked/updated over time) can
# never race with the currently-executing instance re-reading a changed
# file.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

bash scripts/verify_kernel.sh

run_step() {
  local config="$1"
  local label="$2"
  echo "=== $(date -u +%FT%TZ) starting $label ($config) ==="
  python -m src.classification.run_experiment --config "$config" 2>&1 | tee "logs/${label}-dev.log"
  echo "=== $(date -u +%FT%TZ) finished $label ==="
}

run_step configs/llm_few_shot_random_8_qwen_local.json EXP-003-random
run_step configs/llm_few_shot_curated_8_qwen_local.json EXP-004-curated
run_step configs/llm_reasoning_qwen_local.json EXP-005-reasoning

echo "Chain complete: EXP-003, EXP-004, EXP-005 all finished."
