#! /bin/bash
# SeqKD Multi-task: gpt2-base trained on pseudo-labels from multi-task teacher
# Requires pseudo data to be generated first (see generate_data_seqkd_multitask.sh)
# Adapted for 4x RTX 4090
set -euo pipefail

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/ufile/group_3/zjx/distillm"}
CKPT_NAME="gpt2-base"
CKPT="${BASE_PATH}/checkpoints/gpt2-base/"
TEACHER_CKPT_NAME="xlarge-sft-multitask"
TEACHER_CKPT="${BASE_PATH}/results/gpt2/train/sft_multitask/e10-bs4-lr5e-05-G1-N4-NN1/37220"
# data (pseudo-labeled data from SeqKD generation, needs tokenization)
DATA_DIR="${BASE_PATH}/processed_data/combined/pseudo/sft_multitask/gpt2/"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/10M/"
# hp (env vars override defaults — autopipe agent edits exp.json → worker exports TRAIN_*)
BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
LR=${TRAIN_LR:-0.0005}
GRAD_ACC=${TRAIN_GRADIENT_ACCUMULATION_STEPS:-1}
EPOCHS=${TRAIN_EPOCHS:-20}
EVAL_BATCH_SIZE=8
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/gpt2/train/seqkd_multitask/base_xlarge"
# seed
SEED=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 3000"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio 1.0"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type kd"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

if [ -n "${AUTOPIPE_LOAD_PATH:-}" ]; then
    OPTS+=" --load ${AUTOPIPE_LOAD_PATH}"
fi

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
