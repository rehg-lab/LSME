#!/bin/bash

data_path='/data/DevLearning/ShapeNetCore.v2'
output_path='/home/ant/develop/CRIBpp_generic/shapenet_test'
blender_path='/home/ant/Downloads/blender-2.91.2-linux64/blender'
scene_config_path='/home/ant/develop/CRIBpp_generic/common/shapenet_scene_configs_test'

CUDA_VISIBLE_DEVICES=1
python wrapper.py \
    --start=0 \
    --end=20 \
    --dataset_path=$data_path \
    --output_path=$output_path \
    --blender_path=$blender_path \
    --config_path=$scene_config_path \
    --dataset_type=shapenet 2>&1 | tee datagen_log_modelnet.txt
