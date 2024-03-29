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


GLOBAL_BATCH=1504
MICRO_BATCH=188

cur_data="`date +%m%d`"
cur_time="`date +%H%M`"
lr=1e-5
lr_t=1e-5
lr_eff=1e-5
min_lr=1e-6
alias="deepspeed"
text_encoder="t5"
exp_name="/home/cz/bs/rt_torch/history/$cur_data-$cur_time-$text_encoder-$lr_t-$lr_eff-$alias"

cat <<EOT > $DS_CONFIG
{
    "train_batch_size" : $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH,
    "gradient_accumulation_steps": 2,
    
    "fp16": {
        "enabled": true,
        "initial_scale_power": 12
    },

    "tensorboard": {
        "enabled": true,
        "job_name": "train",
        "output_path": $exp_name
    },

    "wall_clock_breakdown" : true,
    "checkpointing": {
        "enabled": true,
        "save_n_recent_checkpoints": 5,
        "max-ckpt-epochs": 5
    },
}
EOT

export NCCL_DEBUG=warn 

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"

# deepspeed --include "localhost:0,1,2,3,4,5,6,7"  --master_port $DS_PORT /home/cz/bs/rt_torch/train_ds.py \
#     --deepspeed_port $DS_PORT \
#     --micro-batch-size $MICRO_BATCH \
#     --global-batch-size $GLOBAL_BATCH \
#     --log-path $exp_name \
#     --train-iters 500000 \
#     --test-iters 100 \
#     --test-interval 2500 \
#     --save-interval 2500 \
#     --seed 42 \
#     --lr-decay-style "cosine" \
#     --lr $lr \
#     --lr_t $lr_t \
#     --lr_eff $lr_eff \
#     --min-lr 1e-5 \
#     --optimizer "adam" \
#     --adam-beta1 0.9 \
#     --adam-beta2 0.95 \
#     --fp16 True \
#     --batch_size $MICRO_BATCH \
#     --loader_bs 1 \
#     --eval-eps 10 \
#     --eval-timeout 100 \
#     --alias $alias \
#     $ds_args

deepspeed --include "localhost:0,1,2,3"  --master_port $DS_PORT /home/cz/bs/rt_torch/train_ds.py \
    --deepspeed_port $DS_PORT \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --log-path $exp_name \
    --train-iters 500000 \
    --test-iters 100 \
    --test-interval 2500 \
    --save-interval 2500 \
    --seed 42 \
    --lr-decay-style "cosine" \
    --lr $lr \
    --lr_t $lr_t \
    --lr_eff $lr_eff \
    --min-lr $min_lr \
    --optimizer "adam" \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --fp16 True \
    --batch_size $MICRO_BATCH \
    --loader_bs 1 \
    --eval-eps 10 \
    --eval-timeout 100 \
    --alias $alias \
    $ds_args

# deepspeed --include "localhost:4,5,6,7"  --master_port $DS_PORT /home/cz/bs/rt_torch/train_ds.py \
#     --deepspeed_port $DS_PORT \
#     --micro-batch-size $MICRO_BATCH \
#     --global-batch-size $GLOBAL_BATCH \
#     --log-path $exp_name \
#     --train-iters 500000 \
#     --test-iters 100 \
#     --test-interval 2500 \
#     --save-interval 2500 \
#     --seed 42 \
#     --lr-decay-style "cosine" \
#     --lr $lr \
#     --lr_t $lr_t \
#     --lr_eff $lr_eff \
#     --min-lr $min_lr \
#     --optimizer "adam" \
#     --adam-beta1 0.9 \
#     --adam-beta2 0.95 \
#     --fp16 True \
#     --batch_size $MICRO_BATCH \
#     --loader_bs 1 \
#     --eval-eps 10 \
#     --eval-timeout 100 \
#     --alias $alias \
#     $ds_args


