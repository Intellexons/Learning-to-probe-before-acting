#!/bin/bash
# One-time setup: uv sync, HF model download, directory structure
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== 1/3 Installing dependencies with uv ==="
uv sync

echo "=== 2/3 Downloading HF model + tokenizer ==="
# Override which base model gets cached at models/hf_base/iter_000 by setting
# LORA4VR_BASE_MODEL before invoking. Default is Qwen3-4B-Instruct-2507, the
# study's frozen-baseline-validated model (see docs/findings.tex 2026-05-10).
BASE_MODEL="${LORA4VR_BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
echo "  base model: $BASE_MODEL"
uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = '$BASE_MODEL'
save_path = 'models/hf_base/iter_000'

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_path)

print('Downloading model (this may take a while)...')
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.save_pretrained(save_path)
print(f'Saved to {save_path}')
"

echo "=== 3/3 Creating directory structure ==="
mkdir -p data/rollouts data/dpo data/tasks
mkdir -p models/hf_base models/lora

# Seed tasks. When LORA4VR_SEED_TASK is unset, fall back to the legacy
# behavior: seed CartPole only if data/tasks/ is empty, never overwrite.
# When set explicitly (e.g. LORA4VR_SEED_TASK=puck), pick the corresponding
# seed files and *always* overwrite — this is the opt-in path used by the
# GCP wrapper to switch between envs on the same VM.
SEED_TASK="${LORA4VR_SEED_TASK:-}"
case "$SEED_TASK" in
  ""|cartpole)
    TRAIN_SRC="scripts/seed_tasks/train.jsonl"
    EVAL_SRC="scripts/seed_tasks/eval.jsonl"
    ;;
  puck)
    TRAIN_SRC="scripts/seed_tasks/puck_train.jsonl"
    EVAL_SRC="scripts/seed_tasks/puck_eval.jsonl"
    ;;
  *)
    echo "ERROR: unknown LORA4VR_SEED_TASK='$SEED_TASK' (expected: cartpole, puck)" >&2
    exit 1
    ;;
esac

if [ -n "$SEED_TASK" ]; then
  cp "$TRAIN_SRC" data/tasks/train.jsonl
  cp "$EVAL_SRC" data/tasks/eval.jsonl
  echo "Seeded data/tasks/{train,eval}.jsonl from $SEED_TASK (overwrite)"
else
  if [ ! -f data/tasks/train.jsonl ]; then
    cp "$TRAIN_SRC" data/tasks/train.jsonl
    echo "Seeded data/tasks/train.jsonl"
  fi
  if [ ! -f data/tasks/eval.jsonl ]; then
    cp "$EVAL_SRC" data/tasks/eval.jsonl
    echo "Seeded data/tasks/eval.jsonl"
  fi
fi

echo ""
echo "Setup complete. Run the loop with: uv run python scripts/09_run_loop.py --config configs/loop.yaml"
