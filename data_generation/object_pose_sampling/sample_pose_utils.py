import bpy
import numpy as np
from mathutils import Matrix

def get_id_info(path, dataset_type):
    if dataset_type == "modelnet":
        category = path.split("/")[-3]
        obj_id = path.split("/")[-1][:-4]
        return category, obj_id

    if dataset_type == "shapenet":
        category = path.split("/")[-4]
        obj_id = path.split("/")[-3]
        return category, obj_id

    if dataset_type == "toys":
        category = path.split("/")[-3]
        obj_id = path.split("/")[-2]
        return category, obj_id

    if dataset_type == "toys200":
        obj_id = path.split('/')[-1]
        category = ""
        return category, obj_id

def load_obj(scn, path, dataset_type):

    category, obj_name = get_id_info(path, dataset_type)

    if dataset_type == "toys200":
        path = '/'.join(path.split('/')[:-1])
        with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects if name == obj_name]

        for obj in data_to.objects:
            if obj is not None:
                scn.collection.objects.link(obj)

        bpy.data.objects[obj_name].name = obj_name
        obj = bpy.data.objects[obj_name]

        obj.hide_viewport = False
        obj.hide_render = False
        obj.rotation_mode = "XYZ"

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        obj.rotation_euler = (np.radians(90), 0, 0)

        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

    if dataset_type == "toys":

        with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects]

        for obj in data_to.objects:
            if obj is not None:
                scn.collection.objects.link(obj)

        in_blender_name = data_from.objects[0]
        bpy.data.objects[in_blender_name].name = obj_name
        obj = bpy.data.objects[obj_name]
        obj.rotation_mode = "XYZ"

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

    if dataset_type == "modelnet":
        bpy.ops.import_scene.obj(
            filepath=path,
            axis_forward="Y",
            axis_up="Z")

        obj_names = [obj.name for obj in bpy.data.objects]
        blender_obj_name = [x for x in sorted(obj_names) if obj_name in x][-1]
        obj = bpy.data.objects[blender_obj_name]

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

    if dataset_type == "shapenet":
        bpy.ops.import_scene.obj(
            filepath=path,
            axis_forward="Y",
            axis_up="Z")

        obj = [obj for obj in bpy.data.objects if (obj.name != "Camera")][0]
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (np.radians(90), 0, 0)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

def add_rigid_body_property(obj_list, p_type):

    for obj in obj_list:

        obj = bpy.data.objects[obj]
        bpy.context.view_layer.objects.active = None
        bpy.context.view_layer.objects.active = obj

        bpy.ops.rigidbody.object_add()
        bpy.context.object.rigid_body.type = p_type

        bpy.context.scene.rigidbody_world.use_split_impulse = False
        bpy.context.scene.rigidbody_world.time_scale = 2
        bpy.context.scene.rigidbody_world.substeps_per_frame = 10
        bpy.context.scene.rigidbody_world.solver_iterations = 10

        bpy.context.object.rigid_body.collision_shape = 'CONVEX_HULL'
        bpy.context.object.rigid_body.use_margin = True
        bpy.context.object.rigid_body.collision_margin = 0.001

def remove_rigid_body_property(obj_list):

    for obj in obj_list:
        obj = bpy.data.objects[obj]
        bpy.context.view_layer.objects.active = None
        bpy.context.view_layer.objects.active = obj

        bpy.ops.rigidbody.object_remove()

def collect_pose(obj):
    scene = bpy.data.scenes['Scene']
    frame_info = []

    import copy
    for frame in range(scene.frame_start,
                       scene.frame_end,
                       scene.frame_step):

        scene.frame_set(frame)
        frame_info.append(copy.deepcopy(obj.matrix_world))

    return frame_info

def add_plane(coords, size):

    bpy.ops.mesh.primitive_cube_add(
            size=1,
            enter_editmode=False,
            align='WORLD',
            location=(0,0,-0.025),
            scale=(size, size, 0.05))

    bpy.data.objects['Cube'].name = 'Plane'

