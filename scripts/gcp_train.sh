#!/bin/bash
# End-to-end GCP training wrapper.
#
# Chains: launch_vm → forward WANDB_API_KEY from local .env → start training
# in a detached tmux session on the VM → print attach/fetch/destroy commands.
#
# Fire-and-forget by default: the script exits once training has started.
# Pass --wait to block locally, poll the tmux session until it finishes, then
# auto-fetch artifacts and destroy the VM.
#
# Usage:
#   bash scripts/gcp_train.sh <config> <run_name> [--wait]
#
# Examples:
#   bash scripts/gcp_train.sh configs/smoke_test.yaml gcp_smoke
#   bash scripts/gcp_train.sh configs/loop.yaml         my_run
#   bash scripts/gcp_train.sh configs/loop_grpo.yaml    grpo_run --wait
#
# Requirements:
#   - .env in the repo root containing `WANDB_API_KEY=...` (prompted for
#     otherwise).
#   - The VM-lifecycle env vars understood by gcp_launch_vm.sh (LORA4VR_*)
#     are forwarded through; override them the same way you would there.

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <config> <run_name> [--wait]" >&2
  echo "Example: $0 configs/smoke_test.yaml gcp_smoke" >&2
  exit 1
fi

CONFIG="$1"
RUN_NAME="$2"
WAIT_MODE=0
if [ $# -ge 3 ] && [ "$3" = "--wait" ]; then
  WAIT_MODE=1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config '$CONFIG' not found (relative to $REPO_ROOT)" >&2
  exit 1
fi

INSTANCE_NAME="${LORA4VR_INSTANCE:-lora4vr-train-1}"
ZONE="${LORA4VR_ZONE:-europe-west3-a}"
REMOTE_DIR="${LORA4VR_REMOTE_DIR:-/opt/lora4vr}"
TMUX_SESSION="lora4vr-train"
DONE_MARKER="/tmp/lora4vr-${RUN_NAME}.done"
LOG_FILE="$REMOTE_DIR/data/progress/${RUN_NAME}.log"

# --- 1. Resolve WANDB_API_KEY from .env (or prompt) -------------------------

if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
  WANDB_API_KEY="$(grep -E '^WANDB_API_KEY=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2-)"
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  read -r -s -p "WANDB_API_KEY (input hidden): " WANDB_API_KEY
  echo
fi
if [ -z "$WANDB_API_KEY" ]; then
  echo "ERROR: WANDB_API_KEY is empty — can't continue" >&2
  exit 1
fi
echo "[gcp_train] W&B key loaded (${#WANDB_API_KEY} chars)"

# --- 2. Bring the VM up (idempotent) ----------------------------------------

echo "[gcp_train] Calling gcp_launch_vm.sh..."
bash "$SCRIPT_DIR/gcp_launch_vm.sh"

# --- 3. Check the tmux session isn't already running this run ---------------

if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" \
     --command="tmux has-session -t $TMUX_SESSION 2>/dev/null" >/dev/null 2>&1; then
  echo "ERROR: a '$TMUX_SESSION' tmux session is already running on $INSTANCE_NAME." >&2
  echo "       Attach with:" >&2
  echo "         gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- -t tmux attach -t $TMUX_SESSION" >&2
  echo "       Or kill it first:" >&2
  echo "         gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tmux kill-session -t $TMUX_SESSION'" >&2
  exit 1
fi

# --- 4. Start training inside tmux ------------------------------------------

echo "[gcp_train] Starting training on $INSTANCE_NAME (config=$CONFIG run_name=$RUN_NAME)..."

# Ensure any stale done-marker is gone so --wait can block cleanly.
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" \
  --command="rm -f $DONE_MARKER && mkdir -p $REMOTE_DIR/data/progress"

# Build the command to run inside tmux. WANDB_API_KEY is set inline so it
# never lands in ~/.bashrc or on-disk history. The `; touch DONE_MARKER` is
# what --wait polls for. ${PIPESTATUS[0]} captures the python orchestrator's
# real exit code; plain `$?` after the pipe would only see tee's exit code
# (almost always 0), masking crashes and making --wait teardown lie.
REMOTE_CMD="cd $REMOTE_DIR && \
  WANDB_API_KEY='$WANDB_API_KEY' \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/09_run_loop.py \
    --config $CONFIG \
    --run_name $RUN_NAME 2>&1 | tee $LOG_FILE; \
  echo \"exit=\${PIPESTATUS[0]}\" > $DONE_MARKER"

gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" \
  --command="tmux new-session -d -s $TMUX_SESSION \"$REMOTE_CMD\""

echo "[gcp_train] Training started in tmux session '$TMUX_SESSION'."

# --- 5. Fire-and-forget path: print the commands and exit -------------------

if [ "$WAIT_MODE" = "0" ]; then
  cat <<EOF

========================================================
Training is running on $INSTANCE_NAME in tmux '$TMUX_SESSION'.
========================================================

Follow the live log:
  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tail -F $LOG_FILE'

Attach the tmux session directly:
  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- -t tmux attach -t $TMUX_SESSION
  (Ctrl-b d to detach without killing training)

When training is done:
  bash scripts/gcp_fetch_artifacts.sh $RUN_NAME
  bash scripts/gcp_destroy_vm.sh

W&B: logs are being sent to your account — check https://wandb.ai for the run.
EOF
  exit 0
fi

# --- 6. --wait path: block until the marker appears, then teardown ----------

echo "[gcp_train] --wait set: blocking until training finishes."
echo "           Safe to Ctrl-C — training keeps running on the VM; you'll"
echo "           just need to run fetch + destroy by hand."
echo ""

while true; do
  if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" \
       --command="test -f $DONE_MARKER" >/dev/null 2>&1; then
    break
  fi
  sleep 30
done

EXIT_LINE="$(gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" \
               --command="cat $DONE_MARKER" 2>/dev/null || echo "exit=?")"
echo "[gcp_train] Training finished ($EXIT_LINE)."

echo "[gcp_train] Fetching artifacts..."
bash "$SCRIPT_DIR/gcp_fetch_artifacts.sh" "$RUN_NAME"

echo "[gcp_train] Destroying VM..."
bash "$SCRIPT_DIR/gcp_destroy_vm.sh" --yes

echo "[gcp_train] Done."
