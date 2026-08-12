#!/usr/bin/env bash
# Startup guard for the Stage B workflow. Run this FIRST, before activating
# the venv or touching the GPU, at the start of every VM session (fresh
# login, reboot, or resumed session alike).
#
# Why this exists: on 2026-08-12 the VM was rebooted to recover the
# known-good kernel/driver combo, and separately `apt-get install
# python3.10-venv` was found to have pulled in kernel package metadata for
# a newer kernel (6.8.0-1064-azure) sitting on disk but not booted -- an
# unverified kernel can silently break the NVIDIA driver (module version
# mismatch), which would surface as a confusing failure deep inside a
# multi-hour experiment rather than as a clear, early error. See
# EXPERIMENT_LOG.md, "VM restart -- /mnt ephemeral-disk data loss incident
# and recovery."
#
# This script does NOT change or upgrade anything -- it only checks and
# fails loudly if the machine is not on the one verified-working
# kernel/driver combo. If it fails, do not proceed with any GPU-heavy
# work; investigate first (see EXPERIMENT_LOG.md for the grub-reboot
# recovery procedure), and never let anything auto-reboot into the
# newer kernel.
set -euo pipefail

KNOWN_GOOD_KERNEL="6.8.0-1029-azure"

ACTUAL_KERNEL="$(uname -r)"
echo "Kernel:  $ACTUAL_KERNEL (expected: $KNOWN_GOOD_KERNEL)"

if [ "$ACTUAL_KERNEL" != "$KNOWN_GOOD_KERNEL" ]; then
  echo "FATAL: running kernel '$ACTUAL_KERNEL' does not match the verified-working" >&2
  echo "kernel '$KNOWN_GOOD_KERNEL'. Do NOT proceed with GPU work -- an unverified" >&2
  echo "kernel can break the NVIDIA driver in ways that surface as a confusing" >&2
  echo "failure mid-experiment rather than here. If this VM booted into an" >&2
  echo "unexpected kernel (e.g. after an automatic update), grub-reboot back to" >&2
  echo "'$KNOWN_GOOD_KERNEL' and reboot -- see EXPERIMENT_LOG.md for the exact" >&2
  echo "procedure used on 2026-08-12. Do not upgrade the driver/CUDA stack to" >&2
  echo "'fix' this instead." >&2
  exit 1
fi

echo "--- nvidia-smi ---"
if ! nvidia-smi; then
  echo "FATAL: nvidia-smi failed even though the kernel matches '$KNOWN_GOOD_KERNEL'." >&2
  echo "The NVIDIA driver module may not be loaded (try: lsmod | grep nvidia)." >&2
  echo "Do not attempt to fix this by upgrading the driver/CUDA stack -- report it" >&2
  echo "before touching anything." >&2
  exit 1
fi

echo "OK: verified kernel ($KNOWN_GOOD_KERNEL) and working NVIDIA driver."
