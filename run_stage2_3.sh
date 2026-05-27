#!/usr/bin/env bash
set -euo pipefail
#
# Stage 2 & 3 of DistiLLM experiment (Stage 1 already done)
#
TORCHRUN="/anaconda3/envs/llm_train/bin/torchrun"
BASE_PATH="/home/ufile/group_3/zjx/distillm"
MASTER_PORT=2012
GPUS_PER_NODE=4

MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"
DS_CONFIG="${BASE_PATH}/configs/deepspeed/ds_config.json"
TEACHER_CKPT="${BASE_PATH}/results/gpt2/train/sft/e10-bs2-lr5e-05-G1-N4-NN1/17500"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${BASE_PATH}"

run_stage() {
    ${TORCHRUN} ${DISTRIBUTED_ARGS} "${BASE_PATH}/finetune.py" "$@"
}

# ============================================================
# Stage 2: Student Init (gpt2-base) — 3 epochs
# ============================================================
echo "=============================================="
echo "[2/3] Student Init (gpt2-base)"
echo "=============================================="

run_stage \
    --base-path ${BASE_PATH} \
    --model-path ${BASE_PATH}/checkpoints/gpt2-base \
    --ckpt-name gpt2-base \
    --n-gpu ${GPUS_PER_NODE} \
    --data-dir ${DATA_DIR} \
    --num-workers 0 \
    --dev-num 1000 \
    --lr 0.0005 \
    --batch-size 8 \
    --eval-batch-size 32 \
    --gradient-accumulation-steps 1 \
    --warmup-iters 0 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --epochs 3 \
    --max-length 512 \
    --max-prompt-length 256 \
    --do-train \
    --do-valid \
    --eval-gen \
    --save-interval -1 \
    --eval-interval -1 \
    --log-interval 4 \
    --mid-log-num -1 \
    --save ${BASE_PATH}/results/gpt2/train/init \
    --seed 10 \
    --deepspeed \
    --deepspeed_config ${DS_CONFIG} \
    --type lm \
    --do-sample --top-k 0 --top-p 1.0 --temperature 1.0

STUDENT_RUN_DIR="$(ls -dt ${BASE_PATH}/results/gpt2/train/init/e3-* 2>/dev/null | head -n 1)"
STUDENT_CKPT="$(ls -d ${STUDENT_RUN_DIR}/*/ 2>/dev/null | sort -t '/' -k1,1 -V | tail -n 1 | tr -d '/ ')"
echo "[2/3] Student Init done. Checkpoint: ${STUDENT_CKPT}"

# ============================================================
# Stage 3: DistiLLM (gpt2-base ← gpt2-xlarge) — 20 epochs
# ============================================================
echo "=============================================="
echo "[3/3] DistiLLM (0.1B ← 1.5B)"
echo "=============================================="

LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/10M/"

run_stage \
    --base-path ${BASE_PATH} \
    --model-path ${STUDENT_CKPT} \
    --teacher-model-path ${TEACHER_CKPT} \
    --ckpt-name gpt2-base \
    --teacher-ckpt-name gpt2-xlarge \
    --teacher-model-fp16 \
    --n-gpu ${GPUS_PER_NODE} \
    --data-dir "${DATA_DIR}" \
    --lm-data-dir "${LM_DATA_DIR}" \
    --num-workers 4 \
    --dev-num 1000 \
    --lr 0.0005 \
    --batch-size 4 \
    --eval-batch-size 16 \
    --gradient-accumulation-steps 8 \
    --warmup-iters 0 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --epochs 20 \
    --kd-ratio 1.0 \
    --max-length 512 \
    --max-prompt-length 256 \
    --do-train \
    --do-valid \
    --eval-gen \
    --save-interval -1 \
    --eval-interval -1 \
    --log-interval 4 \
    --mid-log-num -1 \
    --save ${BASE_PATH}/results/gpt2/train/distill_0.1B_1.5B \
    --seed 10 \
    --deepspeed \
    --deepspeed_config ${DS_CONFIG} \
    --type adaptive-sfkl \
    --student-gen \
    --gen-num-beams 1 \
    --gen-top-p 1.0 \
    --init-threshold 0.0 \
    --loss-eps 0.1 \
    --capacity 1000 \
    --do-sample --top-k 0 --top-p 1.0 --temperature 1.0

echo "=============================================="
echo "[done] Full experiment finished."
echo "=============================================="
