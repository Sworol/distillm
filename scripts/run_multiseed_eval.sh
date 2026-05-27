#!/bin/bash
# Multi-seed Dolly evaluation for DistiLLM checkpoint
# Seeds 20, 30, 40, 50 (seed 10 already done: rougeL 29.06)

TORCHRUN="/anaconda3/envs/llm_train/bin/torchrun"
BASE_PATH="/home/ufile/group_3/zjx/distillm"
CKPT="${BASE_PATH}/results/gpt2/train/distill_0.1B_1.5B/2180"
SAVE="${BASE_PATH}/results/gpt2/eval_main/"
DS_CONFIG="${BASE_PATH}/configs/deepspeed/ds_config.json"
DATA_DIR="${BASE_PATH}/data/dolly"

for seed in 30 40 50; do
    echo "=== Running eval with seed=$seed at $(date) ==="

    MASTER_PORT=$((21000 + seed))

    CUDA_VISIBLE_DEVICES=3 ${TORCHRUN} \
        --nproc_per_node 1 \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr localhost \
        --master_port ${MASTER_PORT} \
        ${BASE_PATH}/evaluate.py \
        --model-path ${CKPT} \
        --ckpt-name distill_0.1B_1.5B/2180 \
        --model-type gpt2 \
        --n-gpu 1 \
        --data-dir ${DATA_DIR} \
        --data-names dolly \
        --num-workers 0 \
        --dev-num -1 \
        --data-process-workers -1 \
        --json-data \
        --eval-batch-size 16 \
        --max-length 512 \
        --max-prompt-length 256 \
        --do-eval \
        --save ${SAVE} \
        --seed ${seed} \
        --deepspeed \
        --deepspeed_config ${DS_CONFIG} \
        --type eval_main \
        --do-sample \
        --top-k 0 \
        --top-p 1.0 \
        --temperature 1.0

    echo "=== Seed $seed done at $(date) ==="
done

echo ""
echo "=== All seeds done. Results: ==="
for seed in 10 20 30 40 50; do
    LOG="${SAVE}/dolly-512/distill_0.1B_1.5B/2180/${seed}/log.txt"
    if [ -f "$LOG" ]; then
        ROUGE=$(grep "rougeL" "$LOG" | tail -1 | grep -oP "'rougeL': \K[0-9.]+")
        echo "Seed $seed: rougeL = $ROUGE"
    else
        echo "Seed $seed: no results yet"
    fi
done
