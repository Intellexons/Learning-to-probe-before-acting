#!/bin/bash
# Timing run for the branched trainer: a few gradient steps at the sizing of the full run, to
# measure the seconds per step and the peak GPU memory. Not a training run: the adapter is
# deleted, the log and the trainer's per-step record are kept. The commit comes from the
# environment, the job name labels the run:
#   hpc submit lora4vr jobs/hpi/train_timing.sbatch --job-name=train-timing-a30 \
#     --export=ALL,LORA4VR_GIT_COMMIT=$(git rev-parse HEAD)
#SBATCH --job-name=train-timing
#SBATCH --account=aisc
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --time=1:30:00
#SBATCH --signal=B:USR1@120
#SBATCH --nodes=1
#SBATCH --gpus=a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
: "${LORA4VR_GIT_COMMIT:?set with sbatch --export=ALL,LORA4VR_GIT_COMMIT=<hash>; results must name their commit}"
CONFIG="${CONFIG:-configs/timing_grpo_branched_crystallizer.yaml}"
RUN_NAME="timing_${SLURM_JOB_ID}"
echo "job $SLURM_JOB_ID on $(hostname), start $(date -Is), commit $LORA4VR_GIT_COMMIT, config $CONFIG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# Results: $OUT (persistent, in the project) gets the log and the trainer's per-step record
# on every exit: success, failure, or the warning before the time limit.
OUT="$SLURM_SUBMIT_DIR/results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
save() {
  cp "data/progress/$RUN_NAME/"*.json "$OUT/" 2>/dev/null || true
  cp "$CONFIG" "$OUT/" 2>/dev/null || true
  rm -rf "models/lora/$RUN_NAME"
  cp "$SLURM_SUBMIT_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUT/" 2>/dev/null || true
  echo "results in $OUT ($(ls "$OUT" | wc -l) entries) at $(date -Is)"
}
trap save EXIT
trap 'echo "time limit near (or hpc stop): saving and exiting with 99"; exit 99' USR1

# Weights were downloaded on the run node; no hub access from the compute node.
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
uv run --frozen python scripts/05_train_grpo_branched.py --iter 0 --config "$CONFIG" --run_name "$RUN_NAME" &
wait $!

echo "done $(date -Is), ${SECONDS}s"
