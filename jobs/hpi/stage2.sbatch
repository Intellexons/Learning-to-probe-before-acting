#!/bin/bash
# Stage 2: the frozen model, probe run, then the final run. One A30 (24 GB) holds a 4B or an 8B
# model in bf16 for generation; nothing is trained. The model, the sample count and the
# commit come from the environment, the job name labels the run:
#   hpc submit lora4vr jobs/hpi/stage2.sbatch --job-name=stage2-8b \
#     --export=ALL,LORA4VR_GIT_COMMIT=$(git rev-parse HEAD),MODEL_PATH=models/hf_base/qwen3-8b,N_SAMPLES=4
#SBATCH --job-name=stage2
#SBATCH --account=aisc
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --time=8:00:00
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
MODEL_PATH="${MODEL_PATH:-models/hf_base/iter_000}"
N_SAMPLES="${N_SAMPLES:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # e.g. --no-earlier-batches for the ablation arm
echo "job $SLURM_JOB_ID on $(hostname), start $(date -Is), commit $LORA4VR_GIT_COMMIT, model $MODEL_PATH, samples $N_SAMPLES"
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
uv run --frozen python scripts/eval_frozen_baseline.py --turns 2 --n-samples "$N_SAMPLES" \
  --model-path "$MODEL_PATH" --out-dir "$WORK" $EXTRA_ARGS &
wait $!

echo "done $(date -Is), ${SECONDS}s"
