#!/usr/bin/env bash
set -euo pipefail

# Small end-to-end run on limited OpenWebText (e.g. 100K) for quick validation.
# Produces:
# - teacher SFT checkpoint (gpt2-xlarge)
# - student init checkpoint (gpt2-base)
# - distillm checkpoint (0.1B <- 1.5B)
#
# Usage:
#   bash scripts/gpt2/smoke/full_small_0.1B_1.5B.sh /PATH/TO/REPO [MASTER_PORT] [GPU_NUM]
#
# Assumes you already prepared:
# - Dolly bin: processed_data/dolly/full/gpt2/
# - OpenWebText bin: processed_data/openwebtext/gpt2/512/100K/

BASE_PATH=${1-"./"}
MASTER_PORT=${2-2012}
GPUS_PER_NODE=${3-4}

MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${BASE_PATH}"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/100K/"

DS_CONFIG="${BASE_PATH}/configs/deepspeed/ds_config.json"

echo "[1/3] Teacher SFT (gpt2-xlarge)"
python -m torch.distributed.run ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py \
  --base-path ${BASE_PATH} \
  --model-path ${BASE_PATH}/checkpoints/gpt2-xlarge \
  --ckpt-name gpt2-xlarge \
  --n-gpu ${GPUS_PER_NODE} \
  --data-dir ${DATA_DIR} \
  --num-workers 0 \
  --train-num 2000 \
  --dev-num 200 \
  --lr 5e-5 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --warmup-iters 0 \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --clip-grad 1.0 \
  --epochs 1 \
  --max-length 512 \
  --max-prompt-length 256 \
  --do-train \
  --do-valid \
  --eval-gen \
  --save-interval 1 \
  --eval-interval 1 \
  --log-interval 1 \
  --mid-log-num -1 \
  --save ${BASE_PATH}/results/gpt2/train/sft \
  --seed 10 \
  --deepspeed \
  --deepspeed_config ${DS_CONFIG} \
  --type lm \
  --do-sample --top-k 0 --top-p 1.0 --temperature 1.0

TEACHER_RUN_DIR="$(ls -dt ${BASE_PATH}/results/gpt2/train/sft/gpt2-xlarge/* | head -n 1)"
TEACHER_CKPT="${TEACHER_RUN_DIR}/1"
echo "Teacher checkpoint: ${TEACHER_CKPT}"

echo "[2/3] Student init (gpt2-base)"
python -m torch.distributed.run ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py \
  --base-path ${BASE_PATH} \
  --model-path ${BASE_PATH}/checkpoints/gpt2-base \
  --ckpt-name gpt2-base \
  --n-gpu ${GPUS_PER_NODE} \
  --data-dir ${DATA_DIR} \
  --num-workers 0 \
  --train-num 2000 \
  --dev-num 200 \
  --lr 5e-5 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --warmup-iters 0 \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --clip-grad 1.0 \
  --epochs 1 \
  --max-length 512 \
  --max-prompt-length 256 \
  --do-train \
  --do-valid \
  --eval-gen \
  --save-interval 1 \
  --eval-interval 1 \
  --log-interval 1 \
  --mid-log-num -1 \
  --save ${BASE_PATH}/results/gpt2/train/init \
  --seed 10 \
  --deepspeed \
  --deepspeed_config ${DS_CONFIG} \
  --type lm \
  --do-sample --top-k 0 --top-p 1.0 --temperature 1.0

STUDENT_RUN_DIR="$(ls -dt ${BASE_PATH}/results/gpt2/train/init/gpt2-base/* | head -n 1)"
STUDENT_INIT_CKPT="${STUDENT_RUN_DIR}/1"
echo "Student init checkpoint: ${STUDENT_INIT_CKPT}"

echo "[3/3] DistiLLM (0.1B <- 1.5B) with OpenWebText=100K"
python -m torch.distributed.run ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py \
  --base-path ${BASE_PATH} \
  --model-path ${STUDENT_INIT_CKPT} \
  --teacher-model-path ${TEACHER_CKPT} \
  --ckpt-name gpt2-base \
  --teacher-ckpt-name gpt2-xlarge \
  --teacher-model-fp16 \
  --n-gpu ${GPUS_PER_NODE} \
  --data-dir ${DATA_DIR} \
  --lm-data-dir ${LM_DATA_DIR} \
  --num-workers 2 \
  --train-num 2000 \
  --dev-num 200 \
  --lr 5e-5 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --warmup-iters 0 \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --clip-grad 1.0 \
  --epochs 1 \
  --kd-ratio 1.0 \
  --max-length 512 \
  --max-prompt-length 256 \
  --do-train \
  --do-valid \
  --eval-gen \
  --save-interval 1 \
  --eval-interval 1 \
  --log-interval 1 \
  --mid-log-num -1 \
  --save ${BASE_PATH}/results/gpt2/train/distillm_small_0.1B_1.5B \
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

echo "[done] Small end-to-end run finished."
