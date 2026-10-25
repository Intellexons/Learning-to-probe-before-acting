#!/bin/bash
# Stage 2 with a chosen prompt example style (EXAMPLE = full | format | none).
#   hpc submit lora4vr jobs/hpi/stage2_example.sbatch --job-name=stage2-ex-none --export=ALL,EXAMPLE=none
#SBATCH --job-name=stage2-ex
#SBATCH --account=aisc
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --nodes=1
#SBATCH --gpus=a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
EXAMPLE="${EXAMPLE:-full}"
echo "job $SLURM_JOB_ID on $(hostname), start $(date -Is), prompt example style: $EXAMPLE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

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

export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
uv run --frozen python scripts/eval_frozen_baseline.py --turns 2 --n-samples 8 --example "$EXAMPLE" \
  --model-path models/hf_base/iter_000 --out-dir "$WORK" &
wait $!

echo "done $(date -Is), ${SECONDS}s"
