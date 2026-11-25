#!/bin/bash
# Evaluation of one checkpoint of a training run, on its own A30, apart from the training job:
# the base model with the run's first N adapters merged in (the model after N iterations),
# evaluated as the frozen model is in Stage 2 (scripts/eval_frozen_baseline.py: the held-out
# tasks, two replies and the tool), so its numbers stand next to the frozen model's. The frozen
# model takes 66 minutes (job 3458956). The job fails at once if the checkpoint is not on disk
# in full. The run's adapter directory, the checkpoint and the commit come from the environment,
# the job name labels the run:
#   hpc submit lora4vr jobs/hpi/eval_checkpoint.sbatch --job-name=eval-<run>-ckpt04 \
#     --export=ALL,LORA4VR_GIT_COMMIT=$(git rev-parse HEAD),N_ADAPTERS=4,LORA_DIR=<dir>
# where <dir> is, for a run trained by jobs/hpi/train.sbatch,
#   /sc/home/<user>/work/<training directory>/models/lora/<run name>
#SBATCH --job-name=eval-checkpoint
#SBATCH --account=aisc
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --time=3:00:00
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
: "${LORA_DIR:?set LORA_DIR to the adapter directory of the training run (holds iter_000, iter_001, ...)}"
: "${N_ADAPTERS:?set N_ADAPTERS to the checkpoint: the number of iterations whose adapters are merged}"
MODEL_PATH="${MODEL_PATH:-models/hf_base/iter_000}"
N_SAMPLES="${N_SAMPLES:-4}"
TURNS="${TURNS:-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
echo "job $SLURM_JOB_ID on $(hostname), start $(date -Is), commit $LORA4VR_GIT_COMMIT, model $MODEL_PATH, checkpoint $N_ADAPTERS of $LORA_DIR, samples $N_SAMPLES, replies $TURNS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

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

# Weights were downloaded on the run node; no hub access from the compute node.
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
uv run --frozen python scripts/eval_frozen_baseline.py --turns "$TURNS" --n-samples "$N_SAMPLES" \
  --model-path "$MODEL_PATH" --lora-dir "$LORA_DIR" --n-adapters "$N_ADAPTERS" --out-dir "$WORK" $EXTRA_ARGS &
wait $!

echo "done $(date -Is), ${SECONDS}s"
