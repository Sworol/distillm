#!/bin/bash
# Tokenize SeqKD pseudo-labels into binary format for training
# Generates: processed_data/combined/pseudo/sft_multitask/
set -euo pipefail

BASE_PATH="/home/ufile/group_3/zjx/distillm"
GEN_DIR="${BASE_PATH}/results/gpt2/gen/sft_multitask/t1.0-l512"
OUT_DIR="${BASE_PATH}/processed_data/combined/pseudo/sft_multitask"

echo "Tokenizing SeqKD pseudo-labels..."
echo "Input: ${GEN_DIR}/raw.jsonl"
echo "Output: ${OUT_DIR}"

mkdir -p "${OUT_DIR}"

python3 "${BASE_PATH}/tools/process_data_dolly.py" \
    --data-dir "${GEN_DIR}" \
    --processed-data-dir "${OUT_DIR}" \
    --model-path "${BASE_PATH}/checkpoints/gpt2-base" \
    --data-process-workers 32 \
    --dev-num 3000

echo "SeqKD tokenization done."
ls -lh "${OUT_DIR}"
