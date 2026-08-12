#!/usr/bin/env bash
# Pull experiment results and logs back from the Azure VM to the local Mac
# repo. Run this from the repo root on the LOCAL machine, ideally after
# every experiment finishes (not just before a planned shutdown) -- /mnt on
# the VM is Azure's ephemeral resource disk and is NOT guaranteed to
# survive a restart/deallocation (see EXPERIMENT_LOG.md, "VM restart /
# /mnt data loss incident"). Anything only living under
# /mnt/vmadmin/projects/sarcasm/{results,logs,data/llm_cache} and not yet
# pulled here can vanish without warning.
#
# Complements scripts/sync_to_vm.sh (which pushes code/config the other
# direction and deliberately excludes results/ and data/llm_cache/).
set -euo pipefail

LOCAL_PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="vmadmin@20.245.56.28"
REMOTE_PATH="/mnt/vmadmin/projects/sarcasm"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/azure_vm_key}"

echo "Remote: $REMOTE_HOST:$REMOTE_PATH"
echo "Local:  $LOCAL_PROJECT_PATH"

rsync -av \
  -e "ssh -i $SSH_KEY" \
  "$REMOTE_HOST:$REMOTE_PATH/results/" \
  "$LOCAL_PROJECT_PATH/results/"

rsync -av \
  -e "ssh -i $SSH_KEY" \
  "$REMOTE_HOST:$REMOTE_PATH/logs/" \
  "$LOCAL_PROJECT_PATH/logs/" 2>/dev/null || echo "(no logs/ dir on remote yet, skipping)"

echo "Done. Now review with 'git status' and commit results/logs so they"
echo "survive independently of the VM's ephemeral disk."
