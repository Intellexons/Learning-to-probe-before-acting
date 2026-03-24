#!/bin/bash
# One-time setup: uv sync, Ollama model, HF model, llama.cpp
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== 1/5 Installing dependencies with uv ==="
uv sync

echo "=== 2/5 Pulling Ollama model ==="
ollama pull qwen2.5-coder:3b-instruct

echo "=== 3/5 Downloading HF model + tokenizer ==="
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

echo "=== 4/5 Cloning and building llama.cpp ==="
if [ ! -d "tools/llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp tools/llama.cpp
fi
cd tools/llama.cpp
cmake -B build
cmake --build build --config Release
cd "$PROJECT_ROOT"

echo "=== 5/5 Creating directory structure ==="
mkdir -p data/rollouts data/dpo
mkdir -p models/hf_base models/lora models/gguf

echo ""
echo "Setup complete. Run scripts with: uv run python scripts/<script>.py"
