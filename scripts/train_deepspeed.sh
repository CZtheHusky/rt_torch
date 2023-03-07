#!/bin/bash
set -ex


if [ $# -lt 1 ] ; then
    DS_PORT=29502
else
    DS_PORT=$1
fi

if [ $# -lt 2 ] ; then
    GPU_LIST="0"
else
    GPU_LIST=$2
fi
USER_PATH=`echo ~`
BASE_PATH=.

DS_CONFIG=ds_config.json


GLOBAL_BATCH=3040
MICRO_BATCH=190
NCKPTS=10

cat <<EOT > $DS_CONFIG
{
    "train_batch_size" : $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH,
    "gradient_accumulation_steps": 4,
    "steps_per_print": 1000,
    
    "fp16": {
        "enabled": true,
        "initial_scale_power": 12
    },


    "tensorboard": {
        "enabled": false,
        "job_name": "train_all"
    },

    "wall_clock_breakdown" : false,
     "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true,
        "number_checkpoints": $NCKPTS,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    }
}
EOT

export NCCL_DEBUG=warn 

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
cur_data="`date +%m%d`"
cur_time="`date +%H%M`"
exp_name="$cur_data-$cur_time"

# deepspeed --include "localhost:0,1,2,3"  --master_port $DS_PORT /home/cz/bs/rt_torch/train_ds.py \
#     --deepspeed_port $DS_PORT \
#     --micro-batch-size $MICRO_BATCH \
#     --global-batch-size $GLOBAL_BATCH \
#     --exp-name $exp_name \
#     --train-iters 500000 \
#     --test-iters 500 \
#     --test-interval 5000 \
#     --save-interval 5000 \
#     --seed 42 \
#     --lr-decay-style "cosine" \
#     --lr 1e-4 \
#     --lr_t 1e-4 \
#     --lr_eff 5e-5 \
#     --master-rank 3 \
#     --min-lr 1e-5 \
#     --optimizer "adam" \
#     --adam-beta1 0.9 \
#     --adam-beta2 0.95 \
#     --fp16 True \
#     --batch_size 190 \
#     --loader_bs 1 \
#     --alias "deepspeed" \
#     $ds_args

deepspeed --include "localhost:4,5,6,7"  --master_port $DS_PORT /home/cz/bs/rt_torch/train_ds.py \
    --deepspeed_port $DS_PORT \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --exp-name $exp_name \
    --train-iters 500000 \
    --test-iters 500 \
    --test-interval 5000 \
    --save-interval 5000 \
    --seed 42 \
    --lr-decay-style "cosine" \
    --lr 1e-4 \
    --lr_t 1e-4 \
    --lr_eff 1e-5 \
    --master-rank 3 \
    --min-lr 1e-5 \
    --optimizer "adam" \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --fp16 True \
    --batch_size 190 \
    --loader_bs 1 \
    --alias "deepspeed" \
    $ds_args
