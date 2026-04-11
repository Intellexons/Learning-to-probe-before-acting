#!/bin/bash
# Prep-only GCE startup script for LoRA4VR training VMs.
#
# Idempotent — safe to re-run on reboots. Does NOT auto-launch training.
#
# Wiring: pass to `gcloud compute instances create` via
#   --metadata-from-file=startup-script=scripts/gcp_startup.sh
# Optional VM metadata keys:
#   LORA4VR_REPO_URL — HTTPS clone URL (default below)
#   LORA4VR_REPO_REF — branch/tag/sha to check out (default: main)
#
# On success, drops a readiness marker at /var/run/lora4vr-ready. Check
# progress from your workstation with:
#   gcloud compute instances get-serial-port-output <name> --zone=<zone>
# or tail the log on the VM itself at /var/log/lora4vr-startup.log.
#
# Assumes a GCP Deep Learning VM image (e.g. common-cu121) so that the
# NVIDIA driver, CUDA, and git are already installed.

set -euo pipefail

LOG=/var/log/lora4vr-startup.log
touch "$LOG" && chmod 644 "$LOG"
exec >> "$LOG" 2>&1

echo ""
echo "========================================================"
echo "[$(date -Iseconds)] lora4vr startup script begin"
echo "========================================================"

REPO_DIR=/opt/lora4vr
READY_MARKER=/var/run/lora4vr-ready

# Read optional metadata overrides. `curl ... metadata.google.internal` is
# the documented way to read instance metadata from inside a GCE VM.
_metadata() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
    || true
}
REPO_URL="$(_metadata LORA4VR_REPO_URL)"
REPO_REF="$(_metadata LORA4VR_REPO_REF)"
REPO_URL="${REPO_URL:-https://github.com/nima-siboni/LoRA4VR.git}"
REPO_REF="${REPO_REF:-main}"

echo "REPO_URL=$REPO_URL"
echo "REPO_REF=$REPO_REF"

# 1. GPU sanity check
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "WARN: nvidia-smi ran but no GPU visible"
else
  echo "WARN: nvidia-smi not installed — expected a Deep Learning VM image"
fi

# 2. Install uv system-wide (into /usr/local/bin so any login user can use it).
#    The upstream installer drops uv into /root/.local/bin when run as root;
#    we copy it onto PATH afterwards.
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ -x /root/.local/bin/uv ]; then
    install -m 755 /root/.local/bin/uv /usr/local/bin/uv
  fi
fi
uv --version || { echo "ERROR: uv install failed"; exit 1; }

# 3. Ensure git is present (should be on Deep Learning VMs, but be defensive)
if ! command -v git >/dev/null 2>&1; then
  apt-get update && apt-get install -y git
fi

# 4. Clone the repo. If it fails (private repo, wrong URL), log it and keep
#    going — the user can `gcloud compute scp` the repo up manually.
mkdir -p "$REPO_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Cloning $REPO_URL into $REPO_DIR"
  if git clone "$REPO_URL" "$REPO_DIR"; then
    (cd "$REPO_DIR" && git checkout "$REPO_REF" || true)
  else
    echo "WARN: clone failed — repo may be private."
    echo "      Upload the working copy with:"
    echo "        gcloud compute scp --recurse <local-LoRA4VR> <vm>:$REPO_DIR --zone=<zone>"
    echo "      then SSH in and run scripts/00_setup_env.sh manually."
  fi
else
  echo "$REPO_DIR already a git repo — updating"
  (cd "$REPO_DIR" && git fetch origin "$REPO_REF" && git checkout "$REPO_REF" && git pull --ff-only origin "$REPO_REF") || true
fi

# 5. Make the repo writable by any OS Login user (all go into google-sudoers
#    group by default on GCE). This lets the SSH user run `uv sync`, write
#    data/, etc. without sudo.
if [ -d "$REPO_DIR" ]; then
  chgrp -R google-sudoers "$REPO_DIR" 2>/dev/null || true
  chmod -R g+rwX "$REPO_DIR"
  # setgid so new files inherit the group
  find "$REPO_DIR" -type d -exec chmod g+s {} \; 2>/dev/null || true
fi

# 6. If the repo is actually present, run one-time setup (uv sync, HF model
#    download, directory + seed task creation). 00_setup_env.sh is idempotent.
if [ -f "$REPO_DIR/scripts/00_setup_env.sh" ]; then
  echo "Running scripts/00_setup_env.sh..."
  (cd "$REPO_DIR" && bash scripts/00_setup_env.sh)
  # Re-apply permissions after uv created .venv/, models/, data/
  chgrp -R google-sudoers "$REPO_DIR" 2>/dev/null || true
  chmod -R g+rwX "$REPO_DIR"
else
  echo "Skipping 00_setup_env.sh — repo not present yet"
fi

# 7. Drop readiness marker + print next-steps hint for the serial console
date -Iseconds > "$READY_MARKER"
echo ""
echo "========================================================"
echo "[$(date -Iseconds)] lora4vr startup script complete"
echo "========================================================"
echo "Ready marker: $READY_MARKER"
echo ""
echo "Next steps — from your workstation:"
echo "  gcloud compute ssh <instance-name> --zone=<zone>"
echo "  cd $REPO_DIR"
echo "  uv run python scripts/09_run_loop.py --config configs/loop.yaml --run_name <name>"
