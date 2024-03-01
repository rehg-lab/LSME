#!/bin/bash

data_path='/data/TOYS4K_BLEND_FILES_PACKED_V1'
output_path='/home/sstojanov3/develop/CRIBpp_generic/common/toys_poses_canonical'
blender_path='/home/sstojanov3/apps/blender-2.91.0-linux64/blender'
orientation='canonical'
n_samples=16

python wrapper.py \
    --start=0 \
    --end=20 \
    --input_path=$data_path \
    --output_path=$output_path \
    --blender_path=$blender_path \
    --orientation=$orientation \
    --n_samples=$n_samples \
    --dataset_type=toys 2>&1 | tee datagen_log_modelnet.txt
