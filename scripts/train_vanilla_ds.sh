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


GLOBAL_BATCH=440
MICRO_BATCH=55

train_iters=500000
cur_data="`date +%m%d`"
cur_time="`date +%H%M%S`"
lr=1e-4
lr_t=1
lr_eff=1
min_lr=1e-5
heads=8
depth=8
model_dim=768
key_dim=96
model_type="vanilla"
fp16="True"
host="localhost:0,1,2,3,4,5,6,7"
alias="deepspeed-$model_type"
text_encoder="use"
exp_name="/home/cz/bs/rt_torch/history/$cur_data-$cur_time-$text_encoder-$lr-$lr_t-$lr_eff-$depth-$model_dim-$alias"
    
    # "scheduler": {
    #     "type": "CosineAnnealingLR",
    #     "params": {
    #         "eta_min": 1e-5,
    #         "T_max": $train_iters,
    #     }
    # }

cat <<EOT > $DS_CONFIG
{
    "train_batch_size" : $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": $lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-4
        }
    },
    "fp16": {
        "enabled": True,
        "initial_scale_power": 12
    },
    "scheduler": {
        "type": "CosineAnnealingLR",
        "params": {
            "eta_min": $min_lr,
            "T_max": $train_iters,
        }
    }
    "tensorboard": {
        "enabled": true,
        "job_name": "train",
        "output_path": $exp_name
    },

    "wall_clock_breakdown" : true,

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
    --load-dir /home/cz/bs/rt_torch/history/0407-015537-use-1e-4-1-1-8-768-deepspeed-vanilla \
    --train-iters $train_iters \
    --text_encoder $text_encoder \
    --test-iters 100 \
    --test-interval 2500 \
    --save-interval 2500 \
    --seed 42 \
    --depth $depth \
    --model_dim $model_dim \
    --key_dim $key_dim \
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
    --eval-eps 5 \
    --eval-timeout 100 \
    --sub_data "language_table_sim" \
    --alias $alias \
    --model $model_type \
    $ds_args