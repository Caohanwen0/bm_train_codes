CUDA_LAUNCH_BLOCKING=1
MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/root/bm_train_codes"
DATA_PATH="/root/bm_train_codes/masked_output"
SAVE_PATH="/root/bm_train_codes/save"
DATASET_NAME="dedup_wwm"
CONFIG="roberta-base"


OPTS=""
OPTS+=" --vocab-file ${BASE_PATH}/${CONFIG}"
OPTS+=" --model-config ${BASE_PATH}/${CONFIG}"
OPTS+=" --input-dataset ${DATA_PATH}/${DATASET_NAME}/"
OPTS+=" --save ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}"

OPTS+=" --load /root/bm_train_codes/save/roberta-base_dedup_wwm/checkpoints/checkpoint-13749.pt"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style linear"
OPTS+=" --lr-decay-iters 125000"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 10"
OPTS+=" --loss-scale 524288"
OPTS+=" --start-step 13800"
OPTS+=" --batch-size 128"
OPTS+=" --lr 5.6249e-04"
OPTS+=" --save-iters 250"
OPTS+=" --log-iters 50"
# OPTS+=" --train-iters 0"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} train.py ${OPTS}"
echo ${CMD}

mkdir -p ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}

if [[ $NODE_RANK == 0 ]]; then
    ${CMD} 2>&1 | tee ${SAVE_PATH}/${CONFIG}_${DATASET_NAME}/logs_$(date +"%Y_%m_%d_%H_%M_%S").log
else
    ${CMD}
fi
