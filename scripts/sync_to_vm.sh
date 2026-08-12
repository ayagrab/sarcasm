#!/usr/bin/env bash
# Sync the local Stage A/B repository to the Azure VM. Run this from the
# repo root on the local machine (NOT on the VM). Requires SSH access to
# vmadmin@20.245.56.28 to already work -- see EXPERIMENT_LOG.md BLOCKER-4b
# if it doesn't yet.
#
# The project is NOT pushed to origin (all Stage A/B work is uncommitted
# on `main` as of this writing -- see `git status`), so this uses rsync
# rather than a git workflow, per instructions.
set -euo pipefail

LOCAL_PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="vmadmin@20.245.56.28"
REMOTE_PATH="/mnt/vmadmin/projects/sarcasm"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/azure_vm_key}"

echo "Local:  $LOCAL_PROJECT_PATH"
echo "Remote: $REMOTE_HOST:$REMOTE_PATH"

ssh -i "$SSH_KEY" "$REMOTE_HOST" "mkdir -p $REMOTE_PATH"

rsync -av \
  -e "ssh -i $SSH_KEY" \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude 'models' \
  --exclude 'checkpoints' \
  --exclude 'results' \
  --exclude '.pytest_cache' \
  --exclude 'data/llm_cache' \
  "$LOCAL_PROJECT_PATH"/ \
  "$REMOTE_HOST:$REMOTE_PATH/"

echo "Done. results/ was excluded (EXP-001 is small and reproducible --"
echo "re-run 'python -m src.classification.run_experiment --config configs/tfidf.json' on the VM if needed)."
