import random
import time
import weakref
from typing import List, Dict

import carla
import numpy as np
from scipy import spatial

RANDOM_VEHICLE = 'vehicle.*'
RANDOM_PEDESTRIAN = 'walker.pedestrian.*'


def image_to_np(image: carla.Image) -> np.asarray:
    array = np.array(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
    array = array[:, :, :3]  # only RGB
    return array[:, :, ::-1]  # BGR


class CarlaWorld:
    """
    reusable class, containing all information relevant for carla
    """

    def __init__(self, client: carla.Client):
        self.client = client
        self.world: carla.World = client.get_world()
        self.debug = self.world.debug
        self.actors: List[carla.Actor] = []
        self.__kd_tree: spatial.cKDTree = None
        self.__waypoints = []

        self.cameras: Dict[str, carla.Sensor] = {}

        self.__load_waypoints()

    def __load_waypoints(self, distance=1.0) -> None:
        """
        Create a k-d tree of waypoints with a certain distance between them for every lane.

        @param distance: approximate distance between waypoints
        @return: None
        """
        waypoint_locations: List[List[float]] = []
        self.__waypoints = []
        for waypoint in self.world.get_map().generate_waypoints(distance):
            waypoint_locations.append([waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z])
            self.__waypoints.append(waypoint)
        self.__kd_tree = spatial.cKDTree(waypoint_locations)

    def spawn_actor(self, wildcard_pattern: str, spawn_point: carla.Location, rotation: carla.Rotation = carla.Rotation(), simulate_physics: bool = False) -> carla.Actor:
        """
        Spawn (random) actor of `wildcard_pattern` at `spawn_point`. When spawn fails `spawn_point.z` will increase until actor can be spawned.

        @param wildcard_pattern: list of actors matching this pattern will be spawn
        @param spawn_point: locations the actor should be spawned (should not collide with any objects)
        @param rotation: rotation of the spawned actor
        @param simulate_physics: determines whether an actor will be affected by physics or not
        @return: spawned actor
        """
        blueprints = self.world.get_blueprint_library().filter(wildcard_pattern)
        blueprint = random.choice(blueprints)
        actor = None
        while not actor:  # fix for "RuntimeError: Spawn failed because of collision at spawn position"
            transform = carla.Transform(spawn_point, rotation)
            actor = self.world.try_spawn_actor(blueprint, transform)
            spawn_point.z += 0.01
        actor.set_simulate_physics(simulate_physics)
        self.actors.append(actor)
        return actor

    def spawn_actor_relative_to_base_location(self, wildcard_pattern: str, spawn_point: carla.Location, rotation: carla.Rotation = carla.Rotation(), simulate_physics: bool = False,
                                              base_location: carla.Location = carla.Location(0, 0, 0), base_rotation: carla.Rotation = carla.Rotation()) -> carla.Actor:
        spawn_point_absolute = base_location + spawn_point
        rotation_absolute = carla.Rotation(base_rotation.pitch + rotation.pitch, base_rotation.yaw + rotation.yaw, base_rotation.roll + rotation.roll)
        return self.spawn_actor(wildcard_pattern, spawn_point_absolute, rotation_absolute, simulate_physics)

    def destroy_actors(self) -> None:
        """
        destroy all actors in the simulator

        @return: None
        """
        for actor in self.actors:
            actor.destroy()
        self.actors = []  # reset list of actors

    @staticmethod
    def __get_frame(weak_self, sensor_name: str, image: carla.Image, frame_filer: int = 10, semantic_segmentation: bool = False, **kwargs) -> None:
        """
        Wrapper for `__save_frame_to_disk` that stops the camera from capturing after an image has been saved.

        @param weak_self: reference to the original `CarlaWorld`
        @param sensor_name: name of the current sensor
        @param image: image from camera
        @param file_name: file name on disk
        @param frame_filer: number of frames to skip between saved images, only needed to sync semantic_segmentation with rgb camera
        @param semantic_segmentation: determines whether the camera is semantic_segmentation
        """
        self = weak_self()
        if not self:
            raise ValueError('self is not defined!')

        if image.frame % frame_filer == 0:
            if semantic_segmentation:
                image.convert(carla.ColorConverter.CityScapesPalette)
            saved_frame = image_to_np(image)

            if saved_frame is not None:
                camera = self.cameras.get(sensor_name, None)
                camera.image = saved_frame
                if camera:
                    camera.stop()

    @staticmethod
    def __save_single_frame_to_disk(weak_self, sensor_name: str, image: carla.Image, file_name: str, frame_filer: int = 10, semantic_segmentation: bool = False) -> None:
        """
        Wrapper for `__save_frame_to_disk` that stops the camera from capturing after an image has been saved.

        @param weak_self: reference to the original `CarlaWorld`
        @param sensor_name: name of the current sensor
        @param image: image from camera
        @param file_name: file name on disk
        @param frame_filer: number of frames to skip between saved images, only needed to sync semantic_segmentation with rgb camera
        @param semantic_segmentation: determines whether the camera is semantic_segmentation
        """
        self = weak_self()
        if not self:
            raise ValueError('self is not defined!')

        saved_frame = CarlaWorld.__save_frame_to_disk(image, file_name, frame_filer, semantic_segmentation)
        if saved_frame is not None:
            camera = self.cameras.get(sensor_name, None)
            camera.image = saved_frame
            if camera:
                camera.stop()

    @staticmethod
    def __save_frame_to_disk(image: carla.Image, file_name: str, frame_filer: int = 10, semantic_segmentation: bool = False) -> bool:
        """
        Callback function for `camera.listen`. Saves an image every `frame_filer` frame to disk.

        @param image: image from camera
        @param file_name: file name on disk
        @param frame_filer: number of frames to skip between saved images, only needed to sync semantic_segmentation with rgb camera
        @param semantic_segmentation: determines whether the camera is semantic_segmentation
        @return: `True` if frame was saved otherwise `False`
        """
        if image.frame % frame_filer == 0:
            if semantic_segmentation:
                image.save_to_disk(file_name % image.frame, carla.ColorConverter.CityScapesPalette)
            else:
                image.save_to_disk(file_name % image.frame)
            return True
        return False

    def spawn_cameras(self, output_path: str, spawn_point: carla.Location, rotation: carla.Rotation = carla.Rotation(), width: int = 1280, height: int = 720, sensors_names: List[str] = None, save_to_disk: bool = True) -> List[carla.Actor]:
        """
        Spawn cameras (rgb and semantic_segmentation) in scene.

        @param output_path: path to output images
        @param spawn_point: locations the camera should be spawned
        @param rotation: rotation of the spawned camera
        @param width: image width in pixels
        @param height: image height in pixels
        @param sensors_names: list og sensor names e.g. `sensor.camera.rgb` or `sensor.camera.semantic_segmentation`
        """
        if sensors_names is None:
            sensors_names = ['sensor.camera.rgb', 'sensor.camera.semantic_segmentation']

        cameras = []
        camera_spawn_point = carla.Transform(spawn_point, rotation)
        for sensor_name in sensors_names:
            camera_bp = self.world.get_blueprint_library().find(sensor_name)
            camera_bp.set_attribute('image_size_x', f'{width}')
            camera_bp.set_attribute('image_size_y', f'{height}')

            camera = self.world.spawn_actor(camera_bp, camera_spawn_point)
            self.actors.append(camera)

            callback = CarlaWorld.__save_single_frame_to_disk if save_to_disk else CarlaWorld.__get_frame

            self.cameras[sensor_name] = camera
            if 'semantic_segmentation' in sensor_name:
                camera.listen(lambda image, sensor_name=sensor_name: callback(weakref.ref(self), sensor_name, image, file_name=f'{output_path}%06d-semantic_segmentation.png', semantic_segmentation=True))
            else:
                camera.listen(lambda image, sensor_name=sensor_name: callback(weakref.ref(self), sensor_name, image, file_name=f'{output_path}%06d.png'))
            cameras.append(camera)

        return cameras

    def get_nearest_waypoint(self, location: carla.Location) -> carla.Waypoint:
        """
        find the nearest waypoint of `location`

        @param location: position from which to search
        @return: nearest waypoint
        """
        result = self.__kd_tree.query([location.x, location.y, location.z])
        return self.__waypoints[result[1]]

    def get_available_maps(self) -> List[str]:
        """
        Returns a list of strings the maps available on server.

        @return: list of map names
        """
        return [map.replace('/Game/Carla/Maps/', '') for map in self.client.get_available_maps()]

    def load_world(self, map_name: str, timeout: float = 20.) -> None:
        """
        Creates a new world with default settings using `map_name` map. All actors in the current world will be destroyed.

        @param map_name: Name of the map to be used in this world. Accepts both full paths and map names, e.g. '/Game/Carla/Maps/Town01' or 'Town01'.
        @param timeout: The maximum time a network call is allowed before blocking it and raising a timeout exceeded error.
        @return: None
        """

        try:
            self.world: carla.World = self.client.load_world(map_name)  # RuntimeError: failed to connect to newly created map
        except RuntimeError:
            print('CarlaWorld', 'waiting 10s for CARLA simulator to respond')
            time.sleep(10)
        self.world.wait_for_tick(timeout)

    def highlight_waypoints(self, distance: float = 1.0, thickness: float = 0.2, life_time: float = 30, hide_lane_changes: bool = False) -> None:
        """
        Draws lines between all waypoints on the map for `life_time` seconds.

        @param distance: Approximate distance between waypoints.
        @param thickness: Density of the line.
        @param life_time: Lifespan in seconds for the shape. By default it only lasts one frame. Set this to 0 for permanent shapes.
        @param hide_lane_changes: Do not draw lane changes.
        @return: None
        """
        for waypoint in self.world.get_map().generate_waypoints(distance):
            col = carla.Color(0, 255, 255) if waypoint.is_junction else carla.Color(0, 255, 0)
            for next_waypoint in waypoint.next(distance):
                self.debug.draw_line(waypoint.transform.location, next_waypoint.transform.location, thickness=thickness, color=col, life_time=life_time)

            if not hide_lane_changes:
                if waypoint.lane_change & carla.LaneChange.Right:
                    right_lane = waypoint.get_right_lane()
                    if right_lane and right_lane.lane_type == carla.LaneType.Driving:
                        self.debug.draw_line(waypoint.transform.location, right_lane.transform.location, thickness=thickness, color=col, life_time=life_time)
                if waypoint.lane_change & carla.LaneChange.Left:
                    left_lane = waypoint.get_left_lane()
                    if left_lane and left_lane.lane_type == carla.LaneType.Driving:
                        self.debug.draw_line(waypoint.transform.location, left_lane.transform.location, thickness=thickness, color=col, life_time=life_time)


if __name__ == '__main__':
    carla_world: CarlaWorld = None
    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)
        carla_world = CarlaWorld(client)

        carla_world.load_world('Town10HD')
    finally:
        if carla_world:
            carla_world.destroy_actors()
