import math
import time
from typing import Union, List, Tuple

import carla
import numpy as np
from networkx import Graph

from utils.carla_world import CarlaWorld


def calc_position(point_on_lane: float, car_attr: dict, lane_attr: dict) -> dict:
    waypoint_idx_next = max(0, int(math.ceil(point_on_lane)))
    waypoint_idx_last = min(car_attr['point_on_lane_max'], int(math.floor(point_on_lane)))

    weight_next_waypoint = waypoint_idx_next - point_on_lane

    waypoint_last = lane_attr['waypoints'][waypoint_idx_last]
    waypoint_next = lane_attr['waypoints'][waypoint_idx_next]

    position = lane_attr['waypoints'][0]
    for key in position.keys():
        position[key] = (1 - weight_next_waypoint) * waypoint_last[key] + weight_next_waypoint * waypoint_next[key]

    return position


class CarlaRenderer:
    """
    Domain-specific renderer definition, takes a scene graph and returns the corresponding rendered image
    """

    def __init__(self, config: dict, carla_world: CarlaWorld):
        """

        @param config: render configs
        @param carla_world: API to interact with carla
        """
        self.config = config
        self.carla_world = carla_world

        self.carla_world.load_world(self.config['attributes']['world_map'])

    def render(self, graphs: Union[Graph, List[Graph]]) -> List[Tuple[np.ndarray, List[dict]]]:
        """
        Render a batch of graphs to their corresponding images

        @param graphs: scene graphs to render
        @return: list of images and labels
        """
        if not isinstance(graphs, list):
            graphs = [graphs]
        return [self._render(graph) for graph in graphs]

    def _render(self, graph: Graph) -> Tuple[np.ndarray, List[dict]]:
        """
        renders a single graph into the corresponding image

        @param graph: scene graph to render
        @return:
        """
        out_img, labels = None, []
        parent_node_dict = {destination: origin for origin, destination in graph.edges}

        class_idx_dict = {}
        for key, value in graph._node.items():
            class_idx_dict.setdefault(value['cls'], []).append(key)

        try:
            car_idxes = class_idx_dict['Car']

            for car_idx in car_idxes:
                car_attr = graph.nodes[car_idx]['attr']
                lane_attr = graph.nodes[parent_node_dict[car_idx]]['attr']  # get lane of car

                point_on_lane = car_attr['point_on_lane'] % (car_attr['point_on_lane_max'] + 1)  # value between 0 and point_on_lane_max
                position = calc_position(point_on_lane, car_attr, lane_attr)

                position['yaw'] = (position['yaw'] + car_attr['yaw']) % car_attr['yaw_max']  # yaw relative to waypoint
                print('CarlaRenderer', point_on_lane, position, f"car_attr['yaw'] = {car_attr['yaw']}")

                self.carla_world.spawn_actor(
                    car_attr['car_type'],
                    carla.Location(x=position['loc_x'], y=position['loc_y'], z=position['loc_z']),
                    carla.Rotation(pitch=position['pitch'], yaw=position['yaw'], roll=position['roll'])
                )

                labels.append({'obj_class': 'car', 'yaw': position['yaw'], 'point_on_lane': point_on_lane})

            camera_idx = class_idx_dict['Camera'][0]
            camera_attr = graph.nodes[camera_idx]['attr']
            cameras = self.carla_world.spawn_cameras(
                camera_attr['output_dir'],
                carla.Location(x=camera_attr['loc_x'], y=camera_attr['loc_y'], z=camera_attr['loc_z']),
                carla.Rotation(pitch=camera_attr['pitch'], yaw=camera_attr['yaw'], roll=camera_attr['roll']),
                width=camera_attr['size'][0], height=camera_attr['size'][1], sensors_names=['sensor.camera.rgb'],
                save_to_disk=False
            )

            labels.append({'obj_class': 'camera', 'z': camera_attr['loc_z']})

            while any(camera.is_listening for camera in self.carla_world.cameras.values()):  # wait until all cameras have stopped listening
                time.sleep(0.1)

            for camera in cameras:
                out_img = camera.image

        finally:
            self.carla_world.destroy_actors()

        return out_img, labels
