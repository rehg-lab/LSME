#!/bin/bash                                   

export MKL_NUM_THREADS=28
export OMP_NUM_THREADS=28
export NUM_NODES=1
export NUM_GPUS_PER_NODE=3
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

GPUS='0,1,2'
CFG='configs/contrastive_global/vispe_moco_ABC.yaml'

export CUDA_VISIBLE_DEVICES=$GPUS

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    --master_port 47771 \
    scripts/train_ddp.py \
    --cfg=$CFG
