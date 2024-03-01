''' filename = "/home/stefan/develop/CRIBpp_generic/object_pose_sampling/sample_pose.py"
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
    parser.add_argument("--orientation", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--n_samples", type=int)

    argv = sys.argv[sys.argv.index("--") + 1 :]
    # debug lines below
    #argv = "--dataset_type toys200 --input_path /home/stefan/ssd_data/objects.blend/wand --orientation=canonical --output_path=/tmp/ --n_samples=1".split(' ')

    args = parser.parse_args(argv)

    scn = bpy.context.scene
    scn.frame_end = 30

    print(args.input_path)
    print(args.dataset_type)
    ### load object
    obj = utils.load_obj(scn, args.input_path, args.dataset_type)
    print(obj)

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

    utils.add_plane((0,0,0), 15.0)
    utils.add_rigid_body_property(['Plane'], 'PASSIVE')

    out_ls = []
    for i in range(args.n_samples):
        print(obj.name)
        if args.orientation == "canonical":
            vertices = np.array([v.co for v in obj.data.vertices])
            obj.location = (0,0,1)

            azim = np.random.uniform(-2*np.pi, 2*np.pi)
            obj.rotation_euler[0] = np.random.uniform(np.radians(-5), np.radians(5))
            obj.rotation_euler[1] = np.random.uniform(np.radians(-5), np.radians(5))
            obj.rotation_euler[2] = azim

        if args.orientation == "3DOF":
            height = np.random.uniform(1,2)
            obj.location = (0,0, height)

            azim = np.random.uniform(-2*np.pi, 2*np.pi)
            elev = np.random.uniform(-2*np.pi, 2*np.pi)
            tilt = np.random.uniform(-2*np.pi, 2*np.pi)

            obj.rotation_euler[0] = azim
            obj.rotation_euler[1] = elev
            obj.rotation_euler[2] = tilt

        bpy.context.view_layer.update()

        utils.add_rigid_body_property([obj.name], "ACTIVE")

        poses =  utils.collect_pose(obj)

        final_pose = np.array(poses[-1])
    
        utils.remove_rigid_body_property([obj.name])
        obj.matrix_world = Matrix(final_pose)

        euler_rot = obj.rotation_euler
        rot_matrix = euler_rot.to_matrix()

        out_dict = {}
        out_dict.update(
            rotation_euler = np.array(euler_rot).tolist(),
            euler_order = str(euler_rot.order),
            rotation_matrix = np.array(rot_matrix).tolist()
        )
        out_ls.append(out_dict)

    out_str = json.dumps(out_ls, indent=True)

    print(obj.matrix_world)
    print(rot_matrix)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(os.path.join(args.output_path, "pose_list.json"), "w") as f:
        f.write(out_str)

if __name__ == "__main__":
    output = main()
