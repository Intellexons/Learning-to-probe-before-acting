#!/bin/bash
# Run eval_frozen_baseline.py on a GCP GPU VM.
#
# Chains: launch_vm → start eval in a detached tmux session → print
# attach/fetch/destroy commands. Fire-and-forget by default.
# Pass --wait to block locally and auto-fetch results when done.
#
# Usage:
#   bash scripts/gcp_eval.sh <run_name> [--wait] [-- <eval_frozen_baseline args>]
#
# Examples:
#   bash scripts/gcp_eval.sh arm_a_001
#   bash scripts/gcp_eval.sh arm_a_001 -- --turns 1 --n-samples 20 --n-seeds 50
#   bash scripts/gcp_eval.sh arm_b_001 --wait -- --turns 2 --n-samples 20 --n-seeds 50
#
# Results land in results/<run_name>/ on the VM; --wait fetches them to
#   ~/Downloads/lora4vr_eval_<run_name>/
#
# VM-lifecycle overrides (same as gcp_train.sh):
#   LORA4VR_INSTANCE, LORA4VR_ZONE, LORA4VR_REMOTE_DIR

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_name> [--wait] [-- <eval_frozen_baseline args>]" >&2
  echo "Example: $0 arm_a_001 -- --turns 1 --n-samples 20 --n-seeds 50" >&2
  exit 1
fi

RUN_NAME="$1"
shift

WAIT_MODE=0
EVAL_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --wait) WAIT_MODE=1; shift ;;
    --) shift; EVAL_ARGS=("$@"); break ;;
    *) EVAL_ARGS+=("$1"); shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

INSTANCE_NAME="${LORA4VR_INSTANCE:-lora4vr-train-1}"
ZONE="${LORA4VR_ZONE:-europe-west3-a}"
REMOTE_DIR="${LORA4VR_REMOTE_DIR:-/opt/lora4vr}"
TMUX_SESSION="lora4vr-eval"
DONE_MARKER="/tmp/lora4vr-eval-${RUN_NAME}.done"
LOG_FILE="$REMOTE_DIR/results/${RUN_NAME}/eval.log"

# --- 1. Bring the VM up (idempotent) ----------------------------------------

echo "[gcp_eval] Calling gcp_launch_vm.sh..."
bash "$SCRIPT_DIR/gcp_launch_vm.sh"

# --- 2. Check the tmux session isn't already running -----------------------

if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" \
     --command="tmux has-session -t $TMUX_SESSION 2>/dev/null" >/dev/null 2>&1; then
  echo "ERROR: a '$TMUX_SESSION' tmux session is already running on $INSTANCE_NAME." >&2
  echo "       Attach with:" >&2
  echo "         gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- -t tmux attach -t $TMUX_SESSION" >&2
  echo "       Or kill it first:" >&2
  echo "         gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tmux kill-session -t $TMUX_SESSION'" >&2
  exit 1
fi

# --- 3. Start eval inside tmux ---------------------------------------------

echo "[gcp_eval] Starting eval on $INSTANCE_NAME (run_name=$RUN_NAME)..."

gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" \
  --command="rm -f $DONE_MARKER && mkdir -p $REMOTE_DIR/results/$RUN_NAME"

EVAL_ARGS_STR="${EVAL_ARGS[*]:-}"
REMOTE_CMD="cd $REMOTE_DIR && \
  uv run python scripts/eval_frozen_baseline.py \
    --out-dir results/$RUN_NAME \
    $EVAL_ARGS_STR \
    2>&1 | tee $LOG_FILE; \
  echo \"exit=\$?\" > $DONE_MARKER"

gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" \
  --command="tmux new-session -d -s $TMUX_SESSION \"$REMOTE_CMD\""

echo "[gcp_eval] Eval started in tmux session '$TMUX_SESSION'."

# --- 4. Fire-and-forget: print commands and exit ---------------------------

if [ "$WAIT_MODE" = "0" ]; then
  cat <<EOF

========================================================
Eval is running on $INSTANCE_NAME in tmux '$TMUX_SESSION'.
========================================================

Follow the live log:
  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tail -F $LOG_FILE'

Attach the tmux session directly:
  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- -t tmux attach -t $TMUX_SESSION
  (Ctrl-b d to detach without stopping eval)

When eval is done, fetch results:
  bash scripts/gcp_fetch_eval.sh $RUN_NAME
  bash scripts/gcp_destroy_vm.sh
EOF
  exit 0
fi

# --- 5. --wait: block until marker appears, then fetch and tear down -------

echo "[gcp_eval] --wait set: blocking until eval finishes."
echo "           Safe to Ctrl-C — eval keeps running on the VM."
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
echo "[gcp_eval] Eval finished ($EXIT_LINE)."

echo "[gcp_eval] Fetching results..."
bash "$SCRIPT_DIR/gcp_fetch_eval.sh" "$RUN_NAME"

echo "[gcp_eval] Destroying VM..."
bash "$SCRIPT_DIR/gcp_destroy_vm.sh" --yes

echo "[gcp_eval] Done."
