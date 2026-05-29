#!/bin/bash
# SeqKD eval: auto-finds SeqKD checkpoint and runs 5-benchmark eval
set -euo pipefail

BASE_PATH="/home/ufile/group_3/zjx/distillm"
SAVE="${BASE_PATH}/results/gpt2/eval_main/"

# Auto-detect SeqKD checkpoint
SEQKD_RUN_DIR="${BASE_PATH}/results/gpt2/train/seqkd_multitask/base_xlarge"
CKPT_DIR=$(ls -dt ${SEQKD_RUN_DIR}/*/*/ 2>/dev/null | head -1)
if [ -z "${CKPT_DIR}" ]; then
    echo "ERROR: No SeqKD checkpoint found in ${SEQKD_RUN_DIR}"
    exit 1
fi

CKPT_NAME="seqkd_multitask/$(basename $(dirname ${CKPT_DIR}))/$(basename ${CKPT_DIR})"

echo "SeqKD eval: CKPT=${CKPT_DIR} NAME=${CKPT_NAME}"
CKPT_PATH="${CKPT_DIR}" CKPT_NAME="${CKPT_NAME}" bash "${BASE_PATH}/scripts/run_eval_baseline.sh" "${1:-10}"
