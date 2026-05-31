#!/usr/bin/env bash
set -euo pipefail
#
# KD Baseline (forward KL): gpt2-base ← gpt2-xlarge multi-task teacher
# Trains on combined data (Dolly+SINST+UINST)
# Adapted for 4x RTX 4090
#
TORCHRUN="/anaconda3/envs/llm_train/bin/torchrun"
BASE_PATH="/home/ufile/group_3/zjx/distillm"
MASTER_PORT=2013
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

# KD: starts from pretrained gpt2-base (not init checkpoint)
STUDENT_CKPT="${BASE_PATH}/checkpoints/gpt2-base"
TEACHER_CKPT="${BASE_PATH}/results/gpt2/train/sft_multitask/e10-bs4-lr5e-05-G1-N4-NN1/37220"

# Hyperparameters (env vars override defaults — autopipe agent edits exp.json → worker exports TRAIN_*)
LR=${TRAIN_LR:-0.0005}
BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
EPOCHS=${TRAIN_EPOCHS:-20}
GRAD_ACC=${TRAIN_GRADIENT_ACCUMULATION_STEPS:-1}
NUM_WORKERS=${TRAIN_NUM_WORKERS:-4}

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${BASE_PATH}"

echo "=============================================="
echo "KD Multi-task Baseline: gpt2-base ← gpt2-xlarge (forward KL)"
echo "Student: ${STUDENT_CKPT}"
echo "Teacher: ${TEACHER_CKPT}"
echo "Data: ${DATA_DIR}"
echo "=============================================="

${TORCHRUN} ${DISTRIBUTED_ARGS} "${BASE_PATH}/finetune.py" \
    --base-path ${BASE_PATH} \
    --model-path ${STUDENT_CKPT} \
    --teacher-model-path ${TEACHER_CKPT} \
    --ckpt-name gpt2-base \
    --teacher-ckpt-name gpt2-xlarge-sft-multitask \
    --teacher-model-fp16 \
    --model-type gpt2 \
    --teacher-model-type gpt2 \
    --n-gpu ${GPUS_PER_NODE} \
    --data-dir "${DATA_DIR}" \
    --num-workers ${NUM_WORKERS} \
    --dev-num 3000 \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    --eval-batch-size 16 \
    --gradient-accumulation-steps ${GRAD_ACC} \
    --warmup-iters 0 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --epochs ${EPOCHS} \
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
    --save ${BASE_PATH}/results/gpt2/train/kd_multitask \
    --seed 10 \
    --deepspeed \
    --deepspeed_config ${DS_CONFIG} \
    --type kd \
    --do-sample \
    --top-k 0 \
    --top-p 1.0 \
    --temperature 1.0 \
    ${AUTOPIPE_LOAD_PATH:+--load "${AUTOPIPE_LOAD_PATH}"}

echo "KD multi-task training done."
