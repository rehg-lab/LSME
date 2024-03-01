'''
    filename = "/home/sstojanov3/develop/CRIBpp_generic/object_pose_sampling/pose_check_load.py"
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
    parser.add_argument("--pose_path", type=str)
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--obj_path", type=str)
    
    '''
    argv = "--pose_path /home/sstojanov3/develop/CRIBpp_generic/common/modelnet_poses_canonical/airplane/airplane_0006/pose_list.json " +\
           "--dataset_type modelnet " +\
           "--obj_path /data/modelnet/modelnet40_aligned_obj/airplane/train/airplane_0006.obj"
    '''

    argv = "--pose_path /home/sstojanov3/develop/CRIBpp_generic/common/toys_poses_canonical/airplane/airplane_002/pose_list.json " +\
           "--dataset_type toys " +\
           "--obj_path /data/TOYS4K_BLEND_FILES_PACKED_V1/airplane/airplane_002/airplane_002.blend"
    
    argv = argv.split(' ')
    print(argv)
    args = parser.parse_args(argv)
    
    with open(args.pose_path, "r") as f:
        poses = json.load(f)

    scn = bpy.context.scene
    cnt = 0
    for row in range(4):

        for col in range(4):

            bpy.context.view_layer.objects.active = None

            for o in bpy.data.objects:
                o.select_set(False)

            obj_path = os.path.join(args.obj_path)
            obj = utils.load_obj(scn, obj_path, args.dataset_type)

            obj.select_set(True)  
            bpy.context.view_layer.objects.active = obj

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
            
            pose_mat = np.eye(4,4)
            rot_mat = np.array(poses[cnt]['rotation_matrix'])
            pose_mat[:3,:3] = rot_mat

            ### calculate z_shift to put object on plane surface
            vertices = np.array([v.co for v in obj.data.vertices])
            vertices_w_co = (rot_mat @ vertices.T).T
            z_shift = np.abs(vertices_w_co[:,2].min())

            ### set location
            pose_mat[0,-1] = row
            pose_mat[1,-1] = col
            pose_mat[2,-1] = z_shift
            
            ### set pose matrix
            obj.matrix_world = Matrix(pose_mat)  

            #obj.location = (row, col, z_shift)

            print(rot_mat)
            print(obj.name)
            print(obj.matrix_world)
            cnt+=1
        
    bpy.context.view_layer.update()

if __name__ == "__main__":
    main()
