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
# study's base model.
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
mkdir -p data/rollouts
mkdir -p models/hf_base models/lora

# No task seeding: the crystallizer is the only environment, and its configs read
# tasks straight from scripts/seed_tasks/crystallizer_*.jsonl.

echo ""
echo "Setup complete. Run the loop with: uv run python scripts/09_run_loop.py --config configs/loop_grpo_branched_crystallizer.yaml"
