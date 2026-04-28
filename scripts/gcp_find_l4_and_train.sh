#!/bin/bash
# Walk L4-bearing zones until one has capacity, then chain into gcp_train.sh.
#
# On stockout (ZONE_RESOURCE_POOL_EXHAUSTED / "does not have enough resources"),
# moves to the next zone. On any non-stockout failure, aborts so we don't leave
# half-created VMs across multiple zones.
#
# Usage:
#   bash scripts/gcp_find_l4_and_train.sh <config> <run_name> [--wait]
#
# Override the zone list:
#   LORA4VR_ZONE_LIST="europe-west4-b europe-west1-c ..." bash scripts/gcp_find_l4_and_train.sh ...

set -uo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <config> <run_name> [--wait]" >&2
  exit 1
fi

CONFIG="$1"
RUN_NAME="$2"
shift 2
EXTRA_ARGS=("$@")

DEFAULT_ZONES=(
  europe-west4-b europe-west4-c europe-west4-a
  europe-west1-b europe-west1-c
  europe-west2-b europe-west2-a
  europe-west3-a europe-west3-b
  europe-west6-b europe-west6-c
)

if [ -n "${LORA4VR_ZONE_LIST:-}" ]; then
  read -r -a ZONES <<< "$LORA4VR_ZONE_LIST"
else
  ZONES=("${DEFAULT_ZONES[@]}")
fi

STOCKOUT_RE='ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|currently unavailable|STOCKOUT|stockout'

for Z in "${ZONES[@]}"; do
  echo ""
  echo "============================================================"
  echo "[find-l4] Attempting zone: $Z"
  echo "============================================================"

  TMP_LOG=$(mktemp -t gcp_find_l4.XXXXXX)

  set +e
  LORA4VR_ZONE="$Z" bash scripts/gcp_train.sh "$CONFIG" "$RUN_NAME" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$TMP_LOG"
  RC=${PIPESTATUS[0]}
  set -e

  if [ "$RC" -eq 0 ]; then
    echo "[find-l4] success in $Z"
    rm -f "$TMP_LOG"
    exit 0
  fi

  if grep -qE "$STOCKOUT_RE" "$TMP_LOG"; then
    echo "[find-l4] stockout in $Z (rc=$RC) — trying next zone"
    rm -f "$TMP_LOG"
    continue
  fi

  echo "[find-l4] non-stockout failure in $Z (rc=$RC) — aborting so we don't leave a partial VM behind"
  echo "[find-l4] error log retained at: $TMP_LOG"
  exit "$RC"
done

echo ""
echo "[find-l4] All ${#ZONES[@]} zones exhausted — no L4 capacity found right now."
exit 2
