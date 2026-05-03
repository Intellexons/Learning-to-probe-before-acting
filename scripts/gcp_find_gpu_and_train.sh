#!/bin/bash
# Walk GPU-bearing zones until one has capacity, then chain into gcp_train.sh.
#
# Supports L4 (default), A100-40GB, A100-80GB, and H100 via --accelerator.
# Generalization of the earlier gcp_find_l4_and_train.sh, which is preserved
# as a thin wrapper for backwards compatibility with logged commands.
#
# On stockout (ZONE_RESOURCE_POOL_EXHAUSTED / "does not have enough resources"),
# moves to the next zone. On any non-stockout failure, aborts so we don't leave
# half-created VMs across multiple zones.
#
# Usage:
#   bash scripts/gcp_find_gpu_and_train.sh <config> <run_name> [--accelerator <type>] [--wait]
#
# Accelerator presets (the --accelerator value):
#   l4          (default)  g2-standard-8     nvidia-l4              24 GiB
#   a100-40gb              a2-highgpu-1g     nvidia-tesla-a100      40 GiB
#   a100-80gb              a2-ultragpu-1g    nvidia-a100-80gb       80 GiB
#   h100                   a3-highgpu-1g     nvidia-h100-80gb       80 GiB
#
# Override the zone list:
#   LORA4VR_ZONE_LIST="europe-west4-b europe-west1-c ..." bash scripts/gcp_find_gpu_and_train.sh ...
# Override machine/accelerator strings directly (skips the preset):
#   LORA4VR_MACHINE_TYPE=... LORA4VR_ACCELERATOR=... bash scripts/gcp_find_gpu_and_train.sh ...

set -uo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <config> <run_name> [--accelerator <type>] [--wait]" >&2
  exit 1
fi

CONFIG="$1"
RUN_NAME="$2"
shift 2

ACCELERATOR_KIND="l4"
PASSTHROUGH=()
while [ $# -gt 0 ]; do
  case "$1" in
    --accelerator)
      ACCELERATOR_KIND="$2"
      shift 2
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

case "$ACCELERATOR_KIND" in
  l4)
    PRESET_MACHINE="g2-standard-8"
    PRESET_ACCEL="type=nvidia-l4,count=1"
    PRESET_ZONES=(
      europe-west4-b europe-west4-c europe-west4-a
      europe-west1-b europe-west1-c
      europe-west2-b europe-west2-a
      europe-west3-a europe-west3-b
      europe-west6-b europe-west6-c
    )
    ;;
  a100-40gb)
    PRESET_MACHINE="a2-highgpu-1g"
    PRESET_ACCEL="type=nvidia-tesla-a100,count=1"
    PRESET_ZONES=(
      europe-west4-a europe-west4-b
      us-central1-a us-central1-b us-central1-c us-central1-f
      us-east1-b us-east1-c
      asia-northeast1-a asia-northeast1-c
    )
    ;;
  a100-80gb)
    PRESET_MACHINE="a2-ultragpu-1g"
    PRESET_ACCEL="type=nvidia-a100-80gb,count=1"
    PRESET_ZONES=(
      europe-west4-a
      us-central1-a us-central1-c
      us-east5-b
      asia-southeast1-c
    )
    ;;
  h100)
    PRESET_MACHINE="a3-highgpu-1g"
    PRESET_ACCEL="type=nvidia-h100-80gb,count=1"
    PRESET_ZONES=(
      europe-west4-b
      us-central1-a us-central1-c
      us-east5-a us-east5-b
      asia-northeast1-b
    )
    ;;
  *)
    echo "Unknown --accelerator '$ACCELERATOR_KIND'. Supported: l4, a100-40gb, a100-80gb, h100." >&2
    exit 1
    ;;
esac

# Env-var overrides take precedence over the preset (matches gcp_launch_vm.sh).
export LORA4VR_MACHINE_TYPE="${LORA4VR_MACHINE_TYPE:-$PRESET_MACHINE}"
export LORA4VR_ACCELERATOR="${LORA4VR_ACCELERATOR:-$PRESET_ACCEL}"

if [ -n "${LORA4VR_ZONE_LIST:-}" ]; then
  read -r -a ZONES <<< "$LORA4VR_ZONE_LIST"
else
  ZONES=("${PRESET_ZONES[@]}")
fi

STOCKOUT_RE='ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|currently unavailable|STOCKOUT|stockout'

echo "[find-gpu] accelerator=$ACCELERATOR_KIND machine=$LORA4VR_MACHINE_TYPE accel=$LORA4VR_ACCELERATOR"

for Z in "${ZONES[@]}"; do
  echo ""
  echo "============================================================"
  echo "[find-gpu] Attempting zone: $Z"
  echo "============================================================"

  TMP_LOG=$(mktemp -t gcp_find_gpu.XXXXXX)

  set +e
  LORA4VR_ZONE="$Z" bash scripts/gcp_train.sh "$CONFIG" "$RUN_NAME" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"} 2>&1 | tee "$TMP_LOG"
  RC=${PIPESTATUS[0]}
  set -e

  if [ "$RC" -eq 0 ]; then
    echo "[find-gpu] success in $Z"
    rm -f "$TMP_LOG"
    exit 0
  fi

  if grep -qE "$STOCKOUT_RE" "$TMP_LOG"; then
    echo "[find-gpu] stockout in $Z (rc=$RC) — trying next zone"
    rm -f "$TMP_LOG"
    continue
  fi

  echo "[find-gpu] non-stockout failure in $Z (rc=$RC) — aborting so we don't leave a partial VM behind"
  echo "[find-gpu] error log retained at: $TMP_LOG"
  exit "$RC"
done

echo ""
echo "[find-gpu] All ${#ZONES[@]} zones exhausted — no $ACCELERATOR_KIND capacity found right now."
exit 2
