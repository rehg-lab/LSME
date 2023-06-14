#!/bin/bash

data_path='/data/DevLearning/toys4k_blend_files'
output_path='/home/ant/develop/CRIBpp_generic/toys_rendering_output_l3'
blender_path='/home/ant/Downloads/blender-2.91.2-linux64/blender'
scene_config_path='/home/ant/develop/CRIBpp_generic/common/toys_scene_configs_l3_test'

python wrapper.py \
    --start=0 \
    --end=20 \
    --dataset_path=$data_path \
    --output_path=$output_path \
    --blender_path=$blender_path \
    --config_path=$scene_config_path \
    --dataset_type=toys 2>&1 | tee datagen_log_modelnet.txt
