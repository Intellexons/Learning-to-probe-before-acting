#!/bin/bash
# Delete the LoRA4VR training VM. Stops all billing for the VM, GPU, and
# the boot disk in one action.
#
# Usage:
#   bash scripts/gcp_destroy_vm.sh
#
# Override defaults via env vars:
#   LORA4VR_INSTANCE — VM name (default: lora4vr-train-1)
#   LORA4VR_ZONE     — GCE zone (default: europe-west3-a)

set -euo pipefail

INSTANCE_NAME="${LORA4VR_INSTANCE:-lora4vr-train-1}"
ZONE="${LORA4VR_ZONE:-europe-west3-a}"

if ! gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" >/dev/null 2>&1; then
  echo "Instance $INSTANCE_NAME not found in zone $ZONE — already deleted?"
  exit 0
fi

echo "About to DELETE instance $INSTANCE_NAME in zone $ZONE."
echo "This permanently destroys the VM and its boot disk."
read -p "Proceed? [y/N] " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
  echo "Aborted."
  exit 0
fi

gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
echo ""
echo "Instance deleted. Billing stopped."
