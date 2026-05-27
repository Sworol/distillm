#!/usr/bin/env bash
set -euo pipefail
#
# Multi-task DistiLLM: gpt2-base ← gpt2-xlarge
# Training on combined data (Dolly + SINST + UINST)
# Adapted for 4x RTX 4090
#
TORCHRUN="/anaconda3/envs/llm_train/bin/torchrun"
BASE_PATH="/home/ufile/group_3/zjx/distillm"
MASTER_PORT=2015
GPUS_PER_NODE=4

MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

DATA_DIR="${BASE_PATH}/processed_data/combined/gpt2/"
DS_CONFIG="${BASE_PATH}/configs/deepspeed/ds_config.json"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${BASE_PATH}"

# Args: $@ = all options
run_stage() {
    ${TORCHRUN} ${DISTRIBUTED_ARGS} "${BASE_PATH}/finetune.py" "$@"
}

# ============================================================
# Stage 1: Teacher SFT (gpt2-xlarge) — 10 epochs on combined data
# 59553 train items / effective bs 8 = 7444 steps/epoch
# Total: ~74K steps
# ============================================================
echo "=============================================="
echo "[1/3] Teacher SFT (gpt2-xlarge) on combined data"
echo "=============================================="

run_stage \
    --base-path ${BASE_PATH} \
    --model-path ${BASE_PATH}/checkpoints/gpt2-xlarge \
    --ckpt-name gpt2-xlarge \
    --n-gpu ${GPUS_PER_NODE} \
    --data-dir ${DATA_DIR} \
    --num-workers 4 \
    --dev-num 3000 \
    --lr 0.00005 \
    --batch-size 2 \
    --eval-batch-size 8 \
    --gradient-accumulation-steps 1 \
    --warmup-iters 0 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --epochs 10 \
    --max-length 512 \
    --max-prompt-length 256 \
    --do-train \
    --do-valid \
    --eval-gen \
    --save-interval -1 \
    --eval-interval -1 \
    --log-interval 4 \
    --mid-log-num -1 \
    --save ${BASE_PATH}/results/gpt2/train/sft_multitask \
    --seed 10 \
    --seed-order 10 \
    --deepspeed \
    --deepspeed_config ${DS_CONFIG} \
    --type lm \
    --do-sample --top-k 0 --top-p 1.0 --temperature 1.0

TEACHER_RUN_DIR="${BASE_PATH}/results/gpt2/train/sft_multitask/e10-bs2-lr5e-05-G1-N4-NN1"
TEACHER_CKPT="$(ls -d ${TEACHER_RUN_DIR}/*/ 2>/dev/null | sort -V | tail -n 1)"
if [ -z "${TEACHER_CKPT}" ]; then
    echo "ERROR: Teacher checkpoint not found in ${TEACHER_RUN_DIR}"
    exit 1
fi
echo "Teacher checkpoint: ${TEACHER_CKPT}"
echo "[1/3] Teacher SFT done. Checkpoint: ${TEACHER_CKPT}"

# ============================================================
# Stage 2: Student Init (gpt2-base) — 3 epochs on combined data
# ============================================================
echo "=============================================="
echo "[2/3] Student Init (gpt2-base) on combined data"
echo "=============================================="

run_stage \
    --base-path ${BASE_PATH} \
    --model-path ${BASE_PATH}/checkpoints/gpt2-base \
    --ckpt-name gpt2-base \
    --n-gpu ${GPUS_PER_NODE} \
    --data-dir ${DATA_DIR} \
    --num-workers 4 \
    --dev-num 3000 \
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
    --save ${BASE_PATH}/results/gpt2/train/init_multitask \
    --seed 10 \
    --deepspeed \
    --deepspeed_config ${DS_CONFIG} \
    --type lm \
    --do-sample --top-k 0 --top-p 1.0 --temperature 1.0

STUDENT_RUN_DIR="$(ls -dt ${BASE_PATH}/results/gpt2/train/init_multitask/e3-* 2>/dev/null | head -n 1)"
STUDENT_CKPT="$(ls -d ${STUDENT_RUN_DIR}/*/ 2>/dev/null | sort -V | tail -n 1)"
if [ -z "${STUDENT_CKPT}" ]; then
    echo "ERROR: Student checkpoint not found in ${STUDENT_RUN_DIR}"
    exit 1
fi
echo "Student checkpoint: ${STUDENT_CKPT}"
echo "[2/3] Student Init done. Checkpoint: ${STUDENT_CKPT}"

# ============================================================
# Stage 3: DistiLLM (gpt2-base ← gpt2-xlarge) — 20 epochs
# ============================================================
echo "=============================================="
echo "[3/3] DistiLLM (0.1B ← 1.5B) on combined data"
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
    --dev-num 3000 \
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
    --save ${BASE_PATH}/results/gpt2/train/distill_multitask \
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
echo "[done] Multi-task experiment finished."
echo "=============================================="
