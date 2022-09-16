#!/usr/bin/env bash

# CONFIG=$1
# GPUS=$2
# PORT=${PORT:-59880}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=59880 \
    $(dirname "$0")/train.py ./local_configs/priorlane/tusimple.py \
     --launcher pytorch ${@:3}
