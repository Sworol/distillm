#! /bin/bash
# SeqKD: Generate pseudo-labels with multi-task teacher on combined data (60K items)
# Runs on 4 GPUs. Expected time: ~12h for 59,553 items at ~3s/item.
set -euo pipefail

MASTER_ADDR=localhost
MASTER_PORT=${2-2115}
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
CKPT_NAME="sft_multitask"
CKPT="${BASE_PATH}/results/gpt2/train/sft_multitask/e10-bs4-lr5e-05-G1-N4-NN1/37220"
# data
DATA_DIR="${BASE_PATH}/processed_data/combined_prompt/gpt2/"
# hp
EVAL_BATCH_SIZE=8
MAX_LENGTH=512
MAX_PROMPT_LENGTH=256
# runtime
SAVE_PATH="${BASE_PATH}/results/gpt2/gen/"


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names combined"
OPTS+=" --num-workers 0"
OPTS+=" --gen-num -1"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed 10"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --type gen"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/generate.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
