#!/bin/bash
# Launch a GCP VM, upload the local working copy, and run 00_setup_env.sh.
# Idempotent: safe to re-run if a step already happened on a previous attempt.
#
# Usage:
#   bash scripts/gcp_launch_vm.sh
#
# Override defaults via env vars (none required for the standard L4 setup):
#   LORA4VR_INSTANCE       — VM name (default: lora4vr-train-1)
#   LORA4VR_ZONE           — GCE zone (default: europe-west3-a)
#   LORA4VR_MACHINE_TYPE   — machine type (default: g2-standard-8)
#   LORA4VR_ACCELERATOR    — GPU spec (default: type=nvidia-l4,count=1)
#   LORA4VR_IMAGE_FAMILY   — Deep Learning VM image family
#   LORA4VR_IMAGE_PROJECT  — image project (default: deeplearning-platform-release)
#   LORA4VR_DISK_SIZE      — boot disk size (default: 100GB)
#   LORA4VR_DISK_TYPE      — boot disk type (default: pd-balanced)
#   LORA4VR_PROVISIONING   — STANDARD (on-demand) or SPOT (default: STANDARD)
#   LORA4VR_REMOTE_DIR     — where to put the repo on the VM (default: /opt/lora4vr)
#   LORA4VR_LOCAL_REPO     — path to the local LoRA4VR checkout (default: pwd)

set -euo pipefail

INSTANCE_NAME="${LORA4VR_INSTANCE:-lora4vr-train-1}"
ZONE="${LORA4VR_ZONE:-europe-west3-a}"
MACHINE_TYPE="${LORA4VR_MACHINE_TYPE:-g2-standard-8}"
ACCELERATOR="${LORA4VR_ACCELERATOR:-type=nvidia-l4,count=1}"
IMAGE_FAMILY="${LORA4VR_IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${LORA4VR_IMAGE_PROJECT:-deeplearning-platform-release}"
DISK_SIZE="${LORA4VR_DISK_SIZE:-100GB}"
DISK_TYPE="${LORA4VR_DISK_TYPE:-pd-balanced}"
PROVISIONING="${LORA4VR_PROVISIONING:-STANDARD}"
REMOTE_DIR="${LORA4VR_REMOTE_DIR:-/opt/lora4vr}"
LOCAL_REPO="${LORA4VR_LOCAL_REPO:-$(pwd)}"

if [ ! -f "$LOCAL_REPO/scripts/gcp_startup.sh" ]; then
  echo "ERROR: $LOCAL_REPO doesn't look like a LoRA4VR checkout" >&2
  echo "       (missing scripts/gcp_startup.sh)" >&2
  echo "       Run from the repo root, or set LORA4VR_LOCAL_REPO=<path>" >&2
  exit 1
fi
cd "$LOCAL_REPO"

echo "[1/6] Checking if instance $INSTANCE_NAME exists in $ZONE..."
if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" >/dev/null 2>&1; then
  echo "      Already exists — skipping create"
else
  echo "[1/6] Creating $INSTANCE_NAME ($MACHINE_TYPE + $ACCELERATOR, $PROVISIONING)..."
  CREATE_FLAGS=(
    --zone="$ZONE"
    --machine-type="$MACHINE_TYPE"
    --accelerator="$ACCELERATOR"
    --image-family="$IMAGE_FAMILY"
    --image-project="$IMAGE_PROJECT"
    --boot-disk-size="$DISK_SIZE"
    --boot-disk-type="$DISK_TYPE"
    --maintenance-policy=TERMINATE
    --scopes=cloud-platform
    --metadata-from-file=startup-script=scripts/gcp_startup.sh
  )
  if [ "$PROVISIONING" = "SPOT" ]; then
    CREATE_FLAGS+=(--provisioning-model=SPOT --instance-termination-action=STOP)
  fi
  gcloud compute instances create "$INSTANCE_NAME" "${CREATE_FLAGS[@]}"
fi

echo "[2/6] Waiting for SSH on $INSTANCE_NAME..."
for i in $(seq 1 30); do
  if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" \
       --command='echo ready' >/dev/null 2>&1; then
    echo "      SSH is up"
    break
  fi
  sleep 5
  if [ "$i" = "30" ]; then
    echo "ERROR: SSH did not come up within 150s" >&2
    exit 1
  fi
done

echo "[3/6] Waiting for the startup script readiness marker..."
for i in $(seq 1 60); do
  if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" \
       --command='test -f /var/run/lora4vr-ready' >/dev/null 2>&1; then
    echo "      /var/run/lora4vr-ready present"
    break
  fi
  sleep 5
  if [ "$i" = "60" ]; then
    echo "ERROR: startup script never finished (no /var/run/lora4vr-ready after 5 min)" >&2
    echo "       Check 'gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE'" >&2
    exit 1
  fi
done

echo "[4/6] Packaging local working copy..."
TARBALL="$(mktemp -t lora4vr.XXXXXX).tar.gz"
trap 'rm -f "$TARBALL"' EXIT
tar -czf "$TARBALL" \
  --exclude='.venv' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='models' \
  --exclude='data' \
  --exclude='tools' \
  --exclude='.claude' \
  --exclude='._*' \
  -C "$LOCAL_REPO" .
SIZE="$(du -h "$TARBALL" | cut -f1)"
echo "      Tarball: $SIZE"

echo "[5/6] Uploading + extracting on the VM..."
gcloud compute scp "$TARBALL" "$INSTANCE_NAME:/tmp/lora4vr.tar.gz" --zone="$ZONE"
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="\
  sudo tar -xzf /tmp/lora4vr.tar.gz -C $REMOTE_DIR && \
  sudo chgrp -R google-sudoers $REMOTE_DIR && \
  sudo chmod -R g+rwX $REMOTE_DIR && \
  rm /tmp/lora4vr.tar.gz && \
  echo 'Upload complete'"

echo "[6/6] Running 00_setup_env.sh on the VM (this may take ~5 min)..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" \
  --command="cd $REMOTE_DIR && bash scripts/00_setup_env.sh"

cat <<EOF

========================================================
VM is ready to train.
========================================================

Smoke test:
  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE
  cd $REMOTE_DIR
  uv run python scripts/09_run_loop.py --config configs/smoke_test.yaml --run_name gcp_smoke

Full run:
  uv run python scripts/09_run_loop.py --config configs/loop.yaml --run_name <name>

Get artifacts back to your Mac:
  bash scripts/gcp_fetch_artifacts.sh <run_name>

Tear down (stops billing):
  bash scripts/gcp_destroy_vm.sh
EOF
