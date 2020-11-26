import json
from typing import List, Dict

import carla
import numpy as np

from utils.carla_world import CarlaWorld


def generate_splice(base_location: carla.Location, length: int, distance: float = 2.0, keep_straight: bool = True, life_time: float = 60.0, reverse: bool = False) -> List[Dict]:
    """
    Generate a splice of size `length` starting at `base_location`.

    @param base_location: start position of the slice
    @param length: number of elements in the slice
    @param distance: The approximate distance where to get the next waypoints.
    @param keep_straight: If true tries keep straight at a junction.
    @param life_time: Lifespan in seconds for the shape. By default it only lasts one frame. Set this to 0 for permanent shapes.
    @return: list of dictionaries with `waypoints` content, ready to dump as JSON
    """
    waypoints: List[carla.Waypoint] = []

    last_waypoint = carla_world.get_nearest_waypoint(base_location)
    while len(waypoints) < length:
        next_waypoints = last_waypoint.next(distance)

        if len(next_waypoints) > 1:  # check which one is the straight one
            all_rotations = []
            for temp_waypoint in next_waypoints:
                total_rotation = 0
                last_rotation = temp_waypoint.transform.rotation.yaw
                for idx in range(10):  # look waypoints into the future
                    temp_waypoint = temp_waypoint.next(2.0)[0]  # always use first waypoint
                    total_rotation += last_rotation - temp_waypoint.transform.rotation.yaw
                    last_rotation = temp_waypoint.transform.rotation.yaw
                all_rotations.append(abs(total_rotation))
            if keep_straight:
                next_waypoint = next_waypoints[np.argmin(all_rotations)]  # use the one with the least change in yaw
            else:
                next_waypoint = next_waypoints[np.argmax(all_rotations)]  # use the one with the most change in yaw
        else:
            next_waypoint = next_waypoints[0]

        col = carla.Color(0, 255, 255)  # if next_waypoint.is_junction else carla.Color(0, 255, 0)
        carla_world.debug.draw_line(last_waypoint.transform.location, next_waypoint.transform.location, thickness=0.2, color=col, life_time=life_time)
        waypoints.append(next_waypoint)
        last_waypoint = next_waypoint

    waypoints_json = waypoints_to_json(waypoints)
    if reverse:
        waypoints_json = list(reversed(waypoints_json))
    return waypoints_json


def waypoints_to_json(waypoints: List[carla.Waypoint]) -> List[Dict]:
    """
    Extract location and rotation of `waypoints` and create a dict, ready to be dumped as JSON.

    @param waypoints: list of waypoints
    @return: list of dictionaries with `waypoints` content, ready to dump as JSON
    """
    parsed_waypoints = []
    for waypoint in waypoints:
        parsed_waypoints.append(waypoint_to_json(waypoint))
    return parsed_waypoints


def waypoint_to_json(waypoint: carla.Waypoint) -> Dict:
    """
    convert single `carla.Waypoint` to JSON

    @param waypoint: carla.Waypoint
    @return: JSON
    """
    location: carla.Location = waypoint.transform.location
    rotation: carla.Rotation = waypoint.transform.rotation

    return {
        'location': {'x': location.x, 'y': location.y, 'z': location.z},
        'rotation': {'pitch': rotation.pitch, 'yaw': rotation.yaw, 'roll': rotation.roll}
    }


def main():
    # carla_world.load_world('Town10HD')
    carla_world.highlight_waypoints(hide_lane_changes=True)

    # region Town 10, Location 1
    waypoints_10_1 = {
        'camera': waypoint_to_json(carla_world.get_nearest_waypoint(carla.Location(x=90, y=66.5, z=0))),
        'splines': [generate_splice(carla.Location(x=95, y=66.5, z=0), 70), generate_splice(carla.Location(x=-39.1, y=76.9, z=0), 70, reverse=True)]
    }
    save_waypoints(waypoints_10_1, 'town_10_1.json')
    # endregion


def save_waypoints(waypoints: Dict, file_name: str) -> None:
    """
    save waypoints as JSON

    @param waypoints: date to be stored
    @param file_name: file_name of JSON file
    @return: None
    """
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(waypoints, f)


if __name__ == '__main__':
    carla_world: CarlaWorld = None
    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        carla_world = CarlaWorld(client)
        main()
    finally:
        if carla_world:
            carla_world.destroy_actors()
