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


GLOBAL_BATCH=2400
MICRO_BATCH=200

cur_data="`date +%m%d`"
cur_time="`date +%H%M%S`"
lr=1e-4
lr_t=1
lr_eff=1
min_lr=1e-5
heads=8
depth=8
model_dim=256
model_type="fusion"
fp16="True"
# host="localhost:1,2,3"
# host="localhost:4,5,6"
host="localhost:0,1,2,3,6,7"
alias="deepspeed-$model_type"
text_encoder="use"
exp_name="/home/cz/bs/rt_torch/history/$cur_data-$cur_time-$text_encoder-$lr-$lr_t-$lr_eff-$depth-$model_dim-$alias"
    

cat <<EOT > $DS_CONFIG
{
    "train_batch_size" : $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "fp16": {
        "enabled": True,
        "initial_scale_power": 12
    },
    "scheduler": {
        "type": "CosineAnnealingLR",
        "params": {
            "eta_min": 1e-5,
            "T_max": $train_iters,
        }
    }
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

deepspeed --include $host  --master_port $DS_PORT /home/cz/bs/rt_torch/train_ds.py \
    --deepspeed_port $DS_PORT \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --log-path $exp_name \
    --train-iters 500000 \
    --test-iters 100 \
    --test-interval 2500 \
    --save-interval 2500 \
    --seed 42 \
    --depth $depth \
    --model_dim $model_dim \
    --lr-decay-style "cosine" \
    --lr $lr \
    --lr_t $lr_t \
    --lr_eff $lr_eff \
    --min-lr $min_lr \
    --optimizer "adam" \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --fp16 $fp16 \
    --batch_size $MICRO_BATCH \
    --loader_bs 1 \
    --eval-eps 10 \
    --eval-timeout 100 \
    --alias $alias \
    --model $model_type \
    $ds_args