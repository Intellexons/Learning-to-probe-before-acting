#!/bin/bash
# Pull training artifacts for a given run from the GCP VM to the local Mac.
#
# Usage:
#   bash scripts/gcp_fetch_artifacts.sh <run_name> [destination]
#
# Example:
#   bash scripts/gcp_fetch_artifacts.sh gcp_smoke_001
#   bash scripts/gcp_fetch_artifacts.sh prod_run_1 /tmp/results
#
# Override defaults via env vars:
#   LORA4VR_INSTANCE   — VM name (default: lora4vr-train-1)
#   LORA4VR_ZONE       — GCE zone (default: europe-west3-a)
#   LORA4VR_REMOTE_DIR — repo path on the VM (default: /opt/lora4vr)

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_name> [destination]" >&2
  echo "Example: $0 gcp_smoke_001" >&2
  exit 1
fi

RUN_NAME="$1"
DEST="${2:-$HOME/Downloads/lora4vr_$RUN_NAME}"
INSTANCE_NAME="${LORA4VR_INSTANCE:-lora4vr-train-1}"
ZONE="${LORA4VR_ZONE:-europe-west3-a}"
REMOTE_DIR="${LORA4VR_REMOTE_DIR:-/opt/lora4vr}"

mkdir -p "$DEST"
echo "Fetching artifacts for run '$RUN_NAME' from $INSTANCE_NAME..."
echo "Destination: $DEST"
echo ""

gcloud compute scp --recurse --zone="$ZONE" \
  "$INSTANCE_NAME:$REMOTE_DIR/models/lora/$RUN_NAME" \
  "$INSTANCE_NAME:$REMOTE_DIR/data/progress/$RUN_NAME" \
  "$INSTANCE_NAME:$REMOTE_DIR/data/rollouts/$RUN_NAME" \
  "$DEST/"

echo ""
echo "Artifacts downloaded to: $DEST"
ls -la "$DEST"
