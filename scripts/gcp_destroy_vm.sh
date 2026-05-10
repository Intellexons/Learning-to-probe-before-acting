#!/bin/bash
# Delete the LoRA4VR training VM. Stops all billing for the VM, GPU, and
# the boot disk in one action.
#
# Usage:
#   bash scripts/gcp_destroy_vm.sh                # interactive — prompts for [y/N]
#   bash scripts/gcp_destroy_vm.sh --yes          # skip the prompt
#   LORA4VR_AUTO_CONFIRM=1 bash scripts/gcp_destroy_vm.sh   # same, via env var
#
# The orchestration scripts (gcp_eval.sh / gcp_train.sh in --wait mode)
# already imply destruction; they pass --yes so the user isn't prompted
# at the end of every run.
#
# Override defaults via env vars:
#   LORA4VR_INSTANCE — VM name (default: lora4vr-train-1)
#   LORA4VR_ZONE     — GCE zone (default: europe-west3-a)

set -euo pipefail

AUTO_CONFIRM="${LORA4VR_AUTO_CONFIRM:-0}"
for arg in "$@"; do
  case "$arg" in
    -y|--yes) AUTO_CONFIRM=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

INSTANCE_NAME="${LORA4VR_INSTANCE:-lora4vr-train-1}"
ZONE="${LORA4VR_ZONE:-europe-west3-a}"

if ! gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" >/dev/null 2>&1; then
  echo "Instance $INSTANCE_NAME not found in zone $ZONE — already deleted?"
  exit 0
fi

echo "About to DELETE instance $INSTANCE_NAME in zone $ZONE."
echo "This permanently destroys the VM and its boot disk."
if [ "$AUTO_CONFIRM" != "1" ]; then
  read -p "Proceed? [y/N] " confirm
  if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
  fi
fi

gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
echo ""
echo "Instance deleted. Billing stopped."
