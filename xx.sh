#! /bin/bash

if [[ $DLS_TASK_NUMBER == 1 ]]; then
    MASTER_ADDR=localhost
    MASTER_PORT=12345
    NNODES=2
    NODE_RANK=0
else
    MASTER_HOST="$BATCH_CUSTOM0_HOSTS"
    MASTER_ADDR="${MASTER_HOST%%:*}"
    MASTER_PORT="${MASTER_HOST##*:}"
    NNODES="$DLS_TASK_NUMBER"
    NODE_RANK="$DLS_TASK_INDEX"
fi

GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/mnt/sfs_turbo/caohanwen/code/bm_train_codes"
DATA_PATH="/mnt/sfs_turbo/caohanwen/data/masked_output"
SAVE_PATH="/mnt/sfs_turbo/caohanwen/data/save"
DATASET_NAME="reddit_twitter_char"
CONFIG="roberta-base"

OPTS=""
OPTS+=" --vocab-file ${BASE_PATH}/${CONFIG}"
OPTS+=" --model-config ${BASE_PATH}/${CONFIG}"
OPTS+=" --input-dataset ${DATA_PATH}/${DATASET_NAME}/"
OPTS+=" --save ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}"

OPTS+=" --load /mnt/sfs_turbo/caohanwen/data/save/roberta-base_reddit_twitter_char/checkpoints/checkpoint-20499.pt"
OPTS+=" --warmup-iters 6000"
OPTS+=" --lr-decay-style linear"
OPTS+=" --lr-decay-iters 250000"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 10"
OPTS+=" --loss-scale 524288"
OPTS+=" --start-step 20500"
OPTS+=" --batch-size 128"
OPTS+=" --lr 7e-04"
OPTS+=" --save-iters 500"
OPTS+=" --log-iters 50"
OPTS+=" --gradient-accumulate 2"
# OPTS+=" --train-iters 0"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} /mnt/sfs_turbo/caohanwen/code/bm_train_codes/train.py ${OPTS}"
echo ${CMD}

mkdir -p ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}

if [[ $NODE_RANK == 0 ]]&&[[ $DLS_TASK_NUMBER == 1 ]]; then
    ${CMD} 2>&1 | tee ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}/logs_$(date +"%Y_%m_%d_%H_%M_%S").log
else
    ${CMD}
fi
