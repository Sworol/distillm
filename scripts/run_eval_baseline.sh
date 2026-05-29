#!/bin/bash
# Generic 5-benchmark eval for baseline experiments
# Usage: CKPT_PATH=/path/to/ckpt CKPT_NAME=name bash scripts/run_eval_baseline.sh [seed]
set -euo pipefail

TORCHRUN="/anaconda3/envs/llm_train/bin/torchrun"
BASE_PATH="/home/ufile/group_3/zjx/distillm"
CKPT="${CKPT_PATH:?CKPT_PATH required}"
CKPT_NAME="${CKPT_NAME:?CKPT_NAME required}"
SAVE="${BASE_PATH}/results/gpt2/eval_main/"
DS_CONFIG="${BASE_PATH}/configs/deepspeed/ds_config.json"
SEED="${1:-10}"

export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH="${BASE_PATH}"

declare -A BENCHMARKS
BENCHMARKS=(
    ["dolly"]="dolly:${BASE_PATH}/data/dolly:-1"
    ["self_inst"]="self-inst:${BASE_PATH}/data/self-inst:-1"
    ["vicuna"]="vicuna:${BASE_PATH}/data/vicuna:-1"
    ["sinst_11_"]="sinst/11_:${BASE_PATH}/data/sinst/11_:-1"
    ["uinst_11_"]="uinst/11_:${BASE_PATH}/data/uinst/11_:10000"
)

RESULTS_FILE="${SAVE}/${CKPT_NAME}_seed${SEED}.txt"
echo "${CKPT_NAME} 5-Benchmark (seed=${SEED})" | tee "$RESULTS_FILE"
echo "Date: $(date)" | tee -a "$RESULTS_FILE"
echo "Checkpoint: ${CKPT}" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

for name in dolly self_inst vicuna sinst_11_ uinst_11_; do
    IFS=':' read -r dname ddir dnum <<< "${BENCHMARKS[$name]}"
    echo "Evaluating: ${dname}" | tee -a "$RESULTS_FILE"

    MASTER_PORT=$((21000 + RANDOM % 1000))

    ${TORCHRUN} \
        --nproc_per_node 1 --nnodes 1 --node_rank 0 \
        --master_addr localhost --master_port ${MASTER_PORT} \
        ${BASE_PATH}/evaluate.py \
        --model-path ${CKPT} \
        --ckpt-name ${CKPT_NAME} \
        --model-type gpt2 \
        --n-gpu 1 \
        --data-dir ${ddir} \
        --data-names ${name} \
        --num-workers 0 \
        --dev-num ${dnum} \
        --data-process-workers -1 \
        --json-data \
        --eval-batch-size 16 \
        --max-length 512 \
        --max-prompt-length 256 \
        --do-eval \
        --save ${SAVE} \
        --seed ${SEED} \
        --deepspeed \
        --deepspeed_config ${DS_CONFIG} \
        --type eval_main \
        --do-sample \
        --top-k 0 --top-p 1.0 --temperature 1.0

    LOG="${SAVE}/${name}-512/${CKPT_NAME}/${SEED}/log.txt"
    if [ -f "$LOG" ]; then
        grep "test | name:" "$LOG" | tail -1 | tee -a "$RESULTS_FILE"
    fi
    echo "" | tee -a "$RESULTS_FILE"
done

echo "Done: ${RESULTS_FILE}"
