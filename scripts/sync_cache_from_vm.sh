#!/usr/bin/env bash
# Pull the per-example LLM disk cache (data/llm_cache/) back from the VM
# to the local Mac, read-only and safe to run at any time (including
# while an experiment is actively running -- it never touches the VM's
# running process, only reads files it has already written).
#
# Why this exists: data/llm_cache/ lives on /mnt (Azure's ephemeral
# resource disk) on the VM and is NOT synced anywhere during a run --
# scripts/sync_to_vm.sh deliberately excludes it going the other
# direction. Both VM-restart incidents on 2026-08-12 (see
# EXPERIMENT_LOG.md) wiped an in-progress experiment's cache along with
# everything else on /mnt, forcing a restart from 0% instead of a resume
# -- real wasted GPU time (not scientific data loss, but avoidable).
#
# Usage: run periodically (e.g. every few minutes) from the local Mac
# while a long LLM experiment is running on the VM. To actually benefit
# from it after a crash, restore the cache to the freshly-recreated VM
# BEFORE relaunching the experiment chain:
#   rsync -av -e "ssh -i ~/.ssh/azure_vm_key" data/llm_cache/ \
#     vmadmin@20.245.56.28:/mnt/vmadmin/projects/sarcasm/data/llm_cache/
set -euo pipefail

LOCAL_PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="vmadmin@20.245.56.28"
REMOTE_PATH="/mnt/vmadmin/projects/sarcasm"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/azure_vm_key}"

rsync -a \
  -e "ssh -i $SSH_KEY -o ConnectTimeout=15" \
  "$REMOTE_HOST:$REMOTE_PATH/data/llm_cache/" \
  "$LOCAL_PROJECT_PATH/data/llm_cache/" 2>/dev/null \
  && echo "$(date -u +%FT%TZ) cache synced: $(ls "$LOCAL_PROJECT_PATH/data/llm_cache/" | wc -l | tr -d ' ') entries" \
  || echo "$(date -u +%FT%TZ) cache sync failed (VM unreachable?)"
