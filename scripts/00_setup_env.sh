#!/bin/bash
# One-time setup: uv sync, HF model download, directory structure
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== 1/3 Installing dependencies with uv ==="
uv sync

echo "=== 2/3 Downloading HF model + tokenizer ==="
uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = 'Qwen/Qwen2.5-Coder-3B-Instruct'
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
mkdir -p data/rollouts data/dpo
mkdir -p models/hf_base models/lora

echo ""
echo "Setup complete. Run the loop with: uv run python scripts/09_run_loop.py --config configs/loop.yaml"
