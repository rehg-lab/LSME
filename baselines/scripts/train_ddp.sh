#!/bin/bash                                   

export MKL_NUM_THREADS=28
export OMP_NUM_THREADS=28
export NUM_NODES=1
export NUM_GPUS_PER_NODE=3
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

GPUS='0,1,2'
CFG='configs/contrastive_global/vispe_moco_ABC.yaml'
# CFG='configs/contrastive_global/trans_cons_ABC.yaml'


# CFG='configs/contrastive_global/vispe_byol_ABC.yaml'

# CFG='configs/contrastive_local/dope_ABC.yaml'
# CFG='configs/contrastive_local/global_from_local_ABC.yaml'

# CKPT="/data/DevLearning/odme_models/logs/2022-09-29-ABC-scene-instance-depth-order-1000-dogt-test/resnet18-dope/2022-09-30-19-10-41/last.pt"

# CKPT="/data/DevLearning/odme_models/logs/2022-11-09-ABC-scene-global-center-crop-full/vispe_moco/2022-11-09-20-08-43_continued_continued/last.pt"
# CKPT="/data/DevLearning/odme_models/logs/2022-11-20-ABC-scene-global-center-crop-full/vispe_byol/2022-11-20-14-49-46/last.pt"
# CKPT="/data/DevLearning/odme_models/logs/2023-02-21-ABC-scene-easier/resnet18-dope/2023-02-21-14-18-46/last.pt"
# CKPT="/data/DevLearning/odme_models/logs/2023-04-20-ABC-scene-global-vispe-trans/trans_high_res/2023-04-24-15-47-00/last.pt"



export CUDA_VISIBLE_DEVICES=$GPUS

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    --master_port 47771 \
    scripts/train_ddp.py \
    --cfg=$CFG
