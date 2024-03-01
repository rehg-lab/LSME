'''
    filename = "/home/sstojanov3/develop/CRIBpp_generic/object_pose_sampling/test_load.py"
    exec(compile(open(filename).read(), filename, 'exec'))
'''

import bpy
import bmesh
import numpy as np
import sys
import os
import json
import argparse
from mathutils import Matrix

### ugly but necessary because of Blender's Python
fpath = bpy.data.filepath
root_path = '/'.join(fpath.split('/')[:-2])
blend_file_dir_path = os.path.join(root_path, "common")
python_file_dir_path = os.path.join(root_path, "object_pose_sampling")

if blend_file_dir_path not in sys.path:
    sys.path.append(blend_file_dir_path)

if python_file_dir_path not in sys.path:
    sys.path.append(python_file_dir_path)

print(sys.path)
import sample_pose_utils as utils

def main():
    ### read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--dataset_type", type=str)

    #argv = sys.argv[sys.argv.index("--") + 1 :]
    #argv = "--dataset_type modelnet --input_path /data/modelnet/modelnet40_aligned_obj/airplane/train/airplane_0003.obj".split(' ')
    #argv = "--dataset_type shapenet --input_path /data/ShapeNet_meshes/ShapeNetCore.v2/02691156/ff77ea82fb4a5f92da9afa637af35064/models/model_normalized.obj".split(' ')
    argv = "--dataset_type toys --input_path /data/TOYS4K_BLEND_FILES_PACKED_V1/airplane/airplane_000/airplane_000.blend".split(' ')
    args = parser.parse_args(argv)

    scn = bpy.context.scene
    scn.frame_end = 250

    ### load object
    obj = utils.load_obj(scn, args.input_path, args.dataset_type)

    # clear normals
    bpy.ops.mesh.customdata_custom_splitnormals_clear()

    # recompute normals
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()

    ### rescaling object to fit in unit cube
    vertices = np.array([v.co for v in obj.data.vertices])
    obj.scale = obj.scale * 0.45 / np.max(np.abs(vertices))
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

if __name__ == "__main__":
    output = main()
