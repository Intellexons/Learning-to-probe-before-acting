#!/bin/bash
# Stage 0: the environment gate. CPU only: hand-written controllers, a random search
# with hill climbing, and the hand-written probe run through the sandbox. The commit
# that produced the result travels in LORA4VR_GIT_COMMIT (the cluster copy has no .git):
#   hpc submit lora4vr jobs/hpi/stage0.sbatch --export=ALL,LORA4VR_GIT_COMMIT=$(git rev-parse HEAD)
#SBATCH --job-name=stage0
#SBATCH --account=aisc
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@120
#SBATCH --nodes=1
#SBATCH --constraint=ARCH:X86
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
: "${LORA4VR_GIT_COMMIT:?set with sbatch --export=ALL,LORA4VR_GIT_COMMIT=<hash>; results must name their commit}"
echo "job $SLURM_JOB_ID on $(hostname), start $(date -Is), commit $LORA4VR_GIT_COMMIT, cpus $SLURM_CPUS_PER_TASK"

# Results: the program writes to $WORK; $OUT (persistent, in the project) gets a
# copy on every exit: success, failure, or the warning before the time limit.
OUT="$SLURM_SUBMIT_DIR/results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
if [ -n "${SLURM_SCRATCH:-}" ]; then WORK="$SLURM_SCRATCH/out"; else WORK="$OUT"; fi
mkdir -p "$OUT" "$WORK"
export OUT WORK
save() {
  if [ "$WORK" != "$OUT" ]; then cp -r "$WORK/." "$OUT/"; fi
  cp "$SLURM_SUBMIT_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUT/" 2>/dev/null || true
  echo "results in $OUT ($(ls "$OUT" | wc -l) entries) at $(date -Is)"
}
trap save EXIT
trap 'echo "time limit near (or hpc stop): saving and exiting with 99"; exit 99' USR1

uv run --frozen python scripts/stage0_gate.py --out "$WORK" --workers "$SLURM_CPUS_PER_TASK" ${EXTRA_ARGS:-} &
wait $!

echo "done $(date -Is), ${SECONDS}s"
