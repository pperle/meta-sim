import json
import sys
from typing import Tuple, List

import bpy
import numpy as np
from bpy.types import SceneObjects
from mathutils import Vector

# TODO: remove ground plane and run in blender

scene_objects = []

GET_ROTATION_BY_OUTER_COORDINATES = False
GET_ROTATION_BY_ROUTEPLANNER = False
GET_ROTATION_BY_VEHICLE_SPAWN_POINT = True

assert GET_ROTATION_BY_OUTER_COORDINATES + GET_ROTATION_BY_ROUTEPLANNER + GET_ROTATION_BY_VEHICLE_SPAWN_POINT <= 1, "Only enable one option to get rotation data"


def get_outer_coordinates_of_obj(scene_object: SceneObjects) -> Tuple[Vector, Vector, Vector, Vector]:
    coords: List[Vector] = [(scene_object.matrix_world @ v.co) for v in scene_object.data.vertices]

    min_x = Vector((float('inf'), 0, 0))
    max_x = Vector((float('-inf'), 0, 0))
    min_y = Vector((0, float('inf'), 0))
    max_y = Vector((0, float('-inf'), 0))

    for coord in coords:  # find outer vales (x and y axis)
        x, y, _ = coord
        if min_x[0] > x:
            min_x = coord
        if max_x[0] < x:
            max_x = coord

        if min_y[1] > y:
            min_y = coord
        if max_y[1] < y:
            max_y = coord

    return min_x, max_x, min_y, max_y


def calc_rotation_of_vectors(min_x: Vector, max_x: Vector, min_y: Vector, max_y: Vector) -> float:
    vector_1 = min_x
    vectors_list = [max_x, min_y, max_y]

    # find nearest vector to vector_1
    vector_dist = float('inf')
    vector_idx = 0
    for idx, vector in enumerate(vectors_list):
        if vector_dist > np.linalg.norm(vector_1 - vector):
            vector_idx = idx
    vector_2 = vectors_list.pop(vector_idx)

    # calc middle vector between the two nearest vectors
    vector_1_2 = (vector_1 + vector_2) / 2
    vector_3_4 = (vectors_list[0] + vectors_list[1]) / 2

    # calc angular
    return np.arctan2(vector_1_2[1] - vector_3_4[1], vector_1_2[0] - vector_3_4[0])


if GET_ROTATION_BY_ROUTEPLANNER or GET_ROTATION_BY_VEHICLE_SPAWN_POINT:
    sys.path.append('/home/pascal/miniconda3/envs/carla_37/lib/python3.7/site-packages/')  # TODO update to local site-packages
    from scipy import spatial

    ext_point_rotations = []
    ext_point_locations = []

    for scene_object in bpy.context.scene.objects:
        if (GET_ROTATION_BY_ROUTEPLANNER and 'RoutePlanner_' in scene_object.name) or (GET_ROTATION_BY_VEHICLE_SPAWN_POINT and 'VehicleSpawnPoint' in scene_object.name):
            ext_point_rotations.append(scene_object.rotation_euler)
            ext_point_locations.append([scene_object.location[0], scene_object.location[1], scene_object.location[2]])

    kd_tree = spatial.cKDTree(ext_point_locations)

for scene_object in bpy.context.scene.objects:
    if 'Road_Road_' in scene_object.name:
        bpy.ops.object.select_all(action='DESELECT')
        scene_object.select_set(state=True)

        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

        rotation = [0., 0., 0.]
        if GET_ROTATION_BY_OUTER_COORDINATES:
            min_x, max_x, min_y, max_y = get_outer_coordinates_of_obj(scene_object)
            angular = calc_rotation_of_vectors(min_x, max_x, min_y, max_y)
            rotation[3] = angular
            print(scene_object.location, np.rad2deg(angular))
        elif GET_ROTATION_BY_ROUTEPLANNER or GET_ROTATION_BY_VEHICLE_SPAWN_POINT:
            location = scene_object.location
            result = kd_tree.query([location[0], location[1], location[2]])
            rotation = ext_point_rotations[result[1]]
            print('KDTree', location, result, ext_point_rotations[result[1]])

        scene_objects.append(
            {
                'name': scene_object.name,
                'location': {'x': scene_object.location[0], 'y': scene_object.location[1] * -1., 'z': scene_object.location[2]},  # * -1. for y axis because y is flipped in UE4
                # 'rotation': {'x': rotation[0], 'y': rotation[1], 'z': rotation[2]}
            }
        )

with open('scene_objects.json', 'w', encoding='utf-8') as f:
    json.dump(scene_objects, f)
