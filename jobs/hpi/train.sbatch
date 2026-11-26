#!/bin/bash
# Training: the loop of scripts/09_run_loop.py on one A30 (24 GB). With the branched config a
# gradient step takes about 4 minutes and peaks at 17 GiB (job 3460791), so the 552 steps of a
# run take about 37 hours. Each iteration saves its adapter under models/lora/<run name>/ and
# its per-step record under data/progress/<run name>/, both in the project directory, so a job
# that is stopped loses the iteration it was in and nothing else.
#
# Submit from a directory that holds the code of one commit and nothing newer (the loop starts
# a new Python process per iteration, so files changed under a running job would change the
# run). The commit comes from the environment, the job name labels the run:
#   hpc submit <dir> jobs/hpi/train.sbatch --job-name=train-f-s0 \
#     --export=ALL,LORA4VR_GIT_COMMIT=$(git rev-parse HEAD)
# To go on after a stop, name the run and the first iteration that is not finished:
#     --export=ALL,LORA4VR_GIT_COMMIT=<hash>,RUN_NAME=<run name>,START_ITER=<k>
#SBATCH --job-name=train
#SBATCH --account=aisc
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --time=2-12:00:00
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
CONFIG="${CONFIG:-configs/loop_grpo_branched_crystallizer.yaml}"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
START_ITER="${START_ITER:-0}"
echo "job $SLURM_JOB_ID on $(hostname), start $(date -Is), commit $LORA4VR_GIT_COMMIT, config $CONFIG, run $RUN_NAME, from iteration $START_ITER"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# Results: $OUT gets the config, the per-step records and the log without the tables of
# completions on every exit: success, failure, or the warning before the time limit. The
# adapters stay in models/lora/$RUN_NAME, the full log in logs/.
OUT="$SLURM_SUBMIT_DIR/results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
LOG="$SLURM_SUBMIT_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
mkdir -p "$OUT"
save() {
  cp "$CONFIG" "$OUT/" 2>/dev/null || true
  cp -r "data/progress/$RUN_NAME" "$OUT/progress" 2>/dev/null || true
  grep -v '^[│╭╰├┌└┏┡┃]' "$LOG" > "$OUT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.compact.out" 2>/dev/null || true
  ls "models/lora/$RUN_NAME" > "$OUT/adapters.txt" 2>/dev/null || true
  echo "results in $OUT ($(ls "$OUT" | wc -l) entries), adapters in models/lora/$RUN_NAME ($(ls "models/lora/$RUN_NAME" 2>/dev/null | wc -l)) at $(date -Is)"
}
trap save EXIT
trap 'echo "time limit near (or hpc stop): saving and exiting with 99"; exit 99' USR1

# Weights were downloaded on the run node; no hub access from the compute node. The loop
# starts each iteration with `uv run`, which must not try to resolve or download either.
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled UV_FROZEN=1 UV_OFFLINE=1
uv run python scripts/09_run_loop.py --config "$CONFIG" --run_name "$RUN_NAME" --start_iter "$START_ITER" &
wait $!

echo "done $(date -Is), ${SECONDS}s"
