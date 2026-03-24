#!/bin/bash
# Create Ollama model from GGUF file.
# Usage: bash scripts/07_ollama_create.sh <iter_num> [quantization] [run_name]
set -euo pipefail

ITER=$1
ITER_PADDED=$(printf "%03d" "$ITER")
QUANT="${2:-Q8_0}"
RUN_NAME="${3:-}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -n "$RUN_NAME" ]; then
    GGUF_DIR="${PROJECT_ROOT}/models/gguf/${RUN_NAME}/iter_${ITER_PADDED}"
else
    GGUF_DIR="${PROJECT_ROOT}/models/gguf/iter_${ITER_PADDED}"
fi
GGUF_FILE="${GGUF_DIR}/model-${QUANT}.gguf"
MODELFILE="${GGUF_DIR}/Modelfile"

if [ ! -f "$GGUF_FILE" ]; then
    echo "ERROR: GGUF file not found at $GGUF_FILE"
    echo "Run scripts/06_convert_to_gguf.sh first."
    exit 1
fi

echo "=== Writing Modelfile ==="
cat > "$MODELFILE" << HEREDOC
FROM ${GGUF_FILE}

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.8
PARAMETER top_p 0.95
HEREDOC

MODEL_NAME="policycoder:iter${ITER_PADDED}"

echo "=== Creating Ollama model: ${MODEL_NAME} ==="
ollama create "$MODEL_NAME" -f "$MODELFILE"

echo "Created Ollama model: ${MODEL_NAME}"
echo "Test with: ollama run ${MODEL_NAME}"
