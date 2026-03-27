#!/bin/bash
# Convert merged HF model to GGUF format.
# Usage: bash scripts/06_convert_to_gguf.sh <iter_num> [quantization] [run_name]
set -euo pipefail

ITER=$1
ITER_PADDED=$(printf "%03d" "$ITER")
QUANT="${2:-Q8_0}"
RUN_NAME="${3:-}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -n "$RUN_NAME" ]; then
    HF_MODEL_PATH="${PROJECT_ROOT}/models/hf_base/${RUN_NAME}/iter_${ITER_PADDED}"
    GGUF_DIR="${PROJECT_ROOT}/models/gguf/${RUN_NAME}/iter_${ITER_PADDED}"
else
    HF_MODEL_PATH="${PROJECT_ROOT}/models/hf_base/iter_${ITER_PADDED}"
    GGUF_DIR="${PROJECT_ROOT}/models/gguf/iter_${ITER_PADDED}"
fi
LLAMA_CPP_DIR="${PROJECT_ROOT}/tools/llama.cpp"

if [ ! -d "$HF_MODEL_PATH" ]; then
    echo "ERROR: HF model not found at $HF_MODEL_PATH"
    exit 1
fi

if [ ! -f "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" ]; then
    echo "ERROR: llama.cpp not found at $LLAMA_CPP_DIR"
    echo "Clone it manually: git clone https://github.com/ggml-org/llama.cpp tools/llama.cpp && cd tools/llama.cpp && cmake -B build && cmake --build build --config Release"
    exit 1
fi

mkdir -p "$GGUF_DIR"

echo "=== Converting HF model to GGUF (FP16) ==="
python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
    "$HF_MODEL_PATH" \
    --outfile "${GGUF_DIR}/model-f16.gguf" \
    --outtype f16

echo "=== Quantizing to ${QUANT} ==="
"${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
    "${GGUF_DIR}/model-f16.gguf" \
    "${GGUF_DIR}/model-${QUANT}.gguf" \
    "$QUANT"

echo "=== Cleaning up FP16 intermediate ==="
rm -f "${GGUF_DIR}/model-f16.gguf"

echo "GGUF model saved to: ${GGUF_DIR}/model-${QUANT}.gguf"
