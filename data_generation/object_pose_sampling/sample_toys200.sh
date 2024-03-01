#!/bin/bash

data_path='/home/stefan/ssd_data/objects.blend'
output_path='/home/stefan/develop/CRIBpp_generic/common/toys200_poses_canonical'
blender_path='/home/stefan/apps/blender-2.93.6-linux-x64/blender'
# canonical or 3DOF - in the latter objects may end up upside down
orientation='canonical'
# how many different times to drop and collect the pose - we may want multiple different poses
n_samples=4

python wrapper.py \
    --start=0 \
    --end=200 \
    --input_path=$data_path \
    --output_path=$output_path \
    --blender_path=$blender_path \
    --orientation=$orientation \
    --n_samples=$n_samples \
    --dataset_type=toys200 2>&1 | tee datagen_log_modelnet.txt
