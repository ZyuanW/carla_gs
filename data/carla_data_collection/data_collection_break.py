# data_collection.py

import carla
import time
import numpy as np
import os
import argparse
import threading
import sys
import json
import random
from datetime import datetime
from PIL import Image  # Added for image processing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=25, help='Data collection frequency (frames per second)')
    parser.add_argument('--output', type=str, default='carla_data', help='Directory to save output data')
    parser.add_argument('--max_f', type=int, default=50, help='Number of frames to collect')
    parser.add_argument('--overtake', action='store_true', help='Enable NPC vehicle to simulate dangerous driving')
    args = parser.parse_args()

    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create output directory if it does not exist
    data_info = {
        'start_time': datetime.now().isoformat(),
        'fps': args.fps,
        'sensors': [],
        'output_dir': os.path.abspath(output_dir)
    }

    # initialize actor list
    actor_list = []

    # Initialize original_settings before try block
    original_settings = None
    frame_number = 0

    try:
        # connect to the Carla server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)  # seconds
        # world = client.get_world()
        # world = client.load_world('Town10')
        # world.unload_map_layer(carla.MapLayer.)
 
        world = client.load_world('Town10HD_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        world.unload_map_layer(carla.MapLayer.Foliage)
        # world = client.load_world('Town06')  # Load Town03 for more challenging driving scenarios

        # Save the original settings
        original_settings = world.get_settings()

        # get the traffic manager and set the global distance to leading vehicle
        traffic_manager = client.get_trafficmanager(8000)  # 8000 is the default TM port
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)  # Set global distance to leading vehicle
        traffic_manager.set_synchronous_mode(True)  # Ensure TM is in synchronous mode

        # get the blueprint library
        blueprint_library = world.get_blueprint_library()

        # spawn the ego vehicle
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        i = np.random.randint(0, len(spawn_points))
        # town10 ## 55(offset(-10,0.0,0.2)),
        # town06 ## 323, 75(offset=15,npc_wait=3,distance=2)
        i = 55
        v_loc = spawn_points[i].location
        # print(f"Spawning ego vehicle at spawn point {i}, {v_loc}")
        spawn_point = spawn_points[i] 
        
        offset_x = -30  # meters
        offset_y = 0.0  # meters
        offset_z = 0.2  # meters
        spawn_point.location.x += offset_x  # adjust the x of the spawn point
        spawn_point.location.y += offset_y  # adjust the y
        spawn_point.location.z += offset_z  # adjust the height
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Ego vehicle spawned at point {i} location:", spawn_point.location)
        actor_list.append(vehicle)  # add vehicle to actor list
        # print("Ego vehicle spawned.")

        # set the vehicle autopilot and speed
        vehicle.set_autopilot(True, traffic_manager.get_port())
        # set the vehicle speed 10% slower than speed limit
        traffic_manager.vehicle_percentage_speed_difference(vehicle, 0)  # Ego vehicle goes 10% slower than speed limit
        # ego vehicle ignores all traffic lights and signs
        traffic_manager.ignore_lights_percentage(vehicle, 100)
        traffic_manager.auto_lane_change(vehicle, False)

        # blueprint for camera and LiDAR sensors
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

        # set camera attributes
        image_width = 1920
        image_height = 1080
        fov = 65  # degrees
        camera_bp.set_attribute('image_size_x', str(image_width))
        camera_bp.set_attribute('image_size_y', str(image_height))
        camera_bp.set_attribute('fov', str(fov))

        # define camera transforms
        camera_transforms = [
            {'id': 0, 'transform': carla.Transform(carla.Location(x=1.5, y=0.0, z=1.6),
                                                   carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))},  # FRONT
            {'id': 1, 'transform': carla.Transform(carla.Location(x=1.5, y=-0.1, z=1.6),
                                                   carla.Rotation(pitch=0.0, yaw=-45.0, roll=0.0))},  # FRONT_LEFT
            {'id': 2, 'transform': carla.Transform(carla.Location(x=1.5, y=0.1, z=1.6),
                                                   carla.Rotation(pitch=0.0, yaw=45.0, roll=0.0))},  # FRONT_RIGHT
            {'id': 3, 'transform': carla.Transform(carla.Location(x=1.5, y=-0.1, z=1.6),
                                                   carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0))},  # SIDE_LEFT
            {'id': 4, 'transform': carla.Transform(carla.Location(x=1.5, y=0.1, z=1.6),
                                                   carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0))},  # SIDE_RIGHT
        ]

        # add camera sensors to data_info
        for cam in camera_transforms:
            data_info['sensors'].append({
                'type': 'camera',
                'id': cam['id'],
                'transform': transform_to_waymo_dict(cam['transform']),
                'attributes': {
                    'image_size_x': image_width,
                    'image_size_y': image_height,
                    'fov': fov
                }
            })

        # create a dictionary to store sensor data and events
        sensor_data = {}
        sensor_events = {}
        cameras = []
     

        # create camera sensors and attach to the vehicle
        for cam in camera_transforms:
            camera = world.spawn_actor(camera_bp, cam['transform'], attach_to=vehicle)
            actor_list.append(camera)  # add camera to actor list
            camera.my_id = cam['id']  # Use custom attribute name
            sensor_data[camera.my_id] = None
            sensor_events[camera.my_id] = threading.Event()
            camera.listen(lambda image, cam_id=camera.my_id: sensor_callback(image, cam_id, sensor_data, sensor_events))
            cameras.append(camera)
       

        # set LiDAR attributes
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('rotation_frequency', str(args.fps))
        lidar_bp.set_attribute('points_per_second', '1000000')
        lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        actor_list.append(lidar)  # add LiDAR sensor to actor list
        sensor_data['lidar'] = None
        sensor_events['lidar'] = threading.Event()
        lidar.listen(lambda point_cloud: lidar_callback(point_cloud, sensor_data, sensor_events))

        # add LiDAR sensor to data_info
        data_info['sensors'].append({
            'type': 'lidar',
            'id': 'lidar_0',
            'transform': transform_to_waymo_dict(lidar_transform),
            'attributes': {
                'range': 100.0,
                'rotation_frequency': args.fps,
                'points_per_second': 1000000
            }
        })

        # create output directories
        images_dir = os.path.join(output_dir, 'images')
        ego_poses_dir = os.path.join(output_dir, 'ego_pose')
        lidar_dir = os.path.join(output_dir, 'lidar')
        concat_dir = os.path.join(output_dir, 'concat')  # Added for concatenated images
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(ego_poses_dir, exist_ok=True)
        os.makedirs(lidar_dir, exist_ok=True)
        os.makedirs(concat_dir, exist_ok=True)  # Ensure 'concat' directory exists
        

        # get the spectator
        spectator = world.get_spectator()

        # set synchronous mode
        # original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.fps
        world.apply_settings(settings)

        # Sync Traffic Manager
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(False)

        # frame_number = 0
        max_frames = args.max_f  # For example, collect 50 frames

        # initialize NPC vehicle
        npc_vehicle = None

        if args.overtake:
            # if NPC vehicle is enabled, try to spawn a vehicle in the left or right lane behind the ego vehicle
            # get the Carla map
            carla_map = world.get_map()

            # try to find the waypoint for the ego vehicle
            ego_vehicle_waypoint = carla_map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)

            # try to find a lane change waypoint for NPC vehicle
            left_lane_waypoint = ego_vehicle_waypoint.get_left_lane()
            right_lane_waypoint = ego_vehicle_waypoint.get_right_lane()

            npc_spawn_waypoint = None
            overtake_direction = None  # 'left' or 'right'

            if left_lane_waypoint and left_lane_waypoint.lane_type == carla.LaneType.Driving:
                print("Left lane is available.")
                # left lane is available, spawn NPC vehicle in left lane
                npc_spawn_waypoint = left_lane_waypoint
                overtake_direction = 'left'
            elif right_lane_waypoint and right_lane_waypoint.lane_type == carla.LaneType.Driving:
                print("Right lane is available.")
                # left lane is not available, spawn NPC vehicle in right lane
                npc_spawn_waypoint = right_lane_waypoint
                overtake_direction = 'right'
            else:
                print("No available lane for NPC vehicle to spawn.")
                npc_spawn_waypoint = None

            if npc_spawn_waypoint:
                
                distance = 2  # meters

                # get a waypoint behind the ego vehicle
                npc_spawn_waypoints = npc_spawn_waypoint.previous(distance)
                
                # # get waypoints in front
                # npc_spawn_waypoints = npc_spawn_waypoint.next(distance)

                if npc_spawn_waypoints:
                    npc_spawn_waypoint = npc_spawn_waypoints[0]  # use the first waypoint in the list
                    npc_spawn_transform = npc_spawn_waypoint.transform

                    # make sure the NPC vehicle is spawned behind the ego vehicle
                    npc_spawn_transform.rotation = ego_vehicle_waypoint.transform.rotation

                    # adjust the spawn location to be slightly behind the ego vehicle
                    npc_spawn_transform.location.x += 7  # adjust the x
                    npc_spawn_transform.location.y += 3.1  # adjust the y
                    npc_spawn_transform.location.z += 0.3  # adjust the height

                    # draw a debug point at the NPC spawn location
                    world.debug.draw_point(npc_spawn_transform.location, size=0.2, color=carla.Color(255,0,0), life_time=10)

                    print("Trying to spawn NPC vehicle at location:", npc_spawn_transform.location)
                    npc_vehicle_bp = blueprint_library.filter('vehicle.audi.tt')[0]
                    npc_vehicle = world.try_spawn_actor(npc_vehicle_bp, npc_spawn_transform)
                    # actor_list.append(npc_vehicle)  # add NPC vehicle to actor list

                    if npc_vehicle is None:
                        print("Failed to spawn NPC vehicle. Please check the spawn location.")
                    else:
                        print("NPC vehicle spawned.")
                        actor_list.append(npc_vehicle)  # add NPC vehicle to actor list

                        # set NPC vehicle autopilot
                        npc_vehicle.set_autopilot(True, traffic_manager.get_port())
                        # set NPC vehicle speed 60% faster than speed limit
                        traffic_manager.vehicle_percentage_speed_difference(npc_vehicle, -10)  # NPC vehicle goes 10% faster than speed limit

                        # NPC vehicle ignores all traffic lights and signs
                        traffic_manager.ignore_lights_percentage(npc_vehicle, 100)
                        traffic_manager.ignore_walkers_percentage(npc_vehicle, 100.0)
                        traffic_manager.ignore_vehicles_percentage(npc_vehicle, 100.0)

                        traffic_manager.auto_lane_change(npc_vehicle, False)

                        # delay the overtaking maneuver by 3 seconds
                        npc_wait = 5
                        def npc_overtake():
                            time.sleep(npc_wait)  # Wait for 3 seconds
                            
                            # enable auto lane change for NPC vehicle
                            npc_vehicle.set_autopilot(False, traffic_manager.get_port())
                            # traffic_manager.auto_lane_change(npc_vehicle, True)

                            overtake_direction = 'break'
                            print("NPC vehicle started break.")
                            if overtake_direction == 'break':
                                control = carla.VehicleControl()
                                control.throttle = 0.0
                                control.brake = 1.0
                                control.steer = 0.0
                                npc_vehicle.apply_control(control)


                            elif overtake_direction == 'left':
                                # if spawned in left lane, overtake and change lane to right (in front of ego vehicle)
                                # traffic_manager.force_lane_change(npc_vehicle, True)  # True for right lane change
                              
                                print("NPC vehicle forced to change lane to right.")
                                control = carla.VehicleControl()
                                control.throttle = 0.5
                                control.steer = 0.1
                                npc_vehicle.apply_control(control)
                                # npc_vehicle.set_autopilot(True, traffic_manager.get_port())
                                # traffic_manager.auto_lane_change(npc_vehicle, False)
                                traffic_manager.force_lane_change(npc_vehicle, True)
                                traffic_manager.force_lane_change(npc_vehicle, True)
                                traffic_manager.force_lane_change(npc_vehicle, True)
                                print("NPC vehicle forced go stright.")
                                # npc_vehicle.apply_control(control)
                                
                            elif overtake_direction == 'right':
                                # if spawned in right lane, overtake and change lane to left (in front of ego vehicle)
                                # traffic_manager.force_lane_change(npc_vehicle, False)  # False for left lane change

                                
                                # control = carla.VehicleControl()
                                # control.throttle = 0.5
                                # control.steer = -0.5
                                print("NPC vehicle forced to change lane to left.")

                                npc_vehicle.set_autopilot(True, traffic_manager.get_port())
                                traffic_manager.force_lane_change(npc_vehicle, False)
                                traffic_manager.force_lane_change(npc_vehicle, False)
                                traffic_manager.force_lane_change(npc_vehicle, False)
                                # npc_vehicle.apply_control(control)
                                print("NPC vehicle forced go stright.")
                            # print("NPC vehicle started overtaking maneuver.")

                        # start a new thread to handle NPC overtaking
                        threading.Thread(target=npc_overtake).start()
                else:
                    print("No spawn waypoints found behind the NPC lane.")
            else:
                print("NPC vehicle not spawned due to unavailable lanes.")


        wait_for_data_collection = 60  # seconds
        # let the vehicles move for {wait for data} seconds before data collection starts
        print(f"wait Vehicles are moving for {wait_for_data_collection} seconds before data collection starts...")
        for _ in range(int(wait_for_data_collection * args.fps)):
            snapshot = world.tick()
            loc = vehicle.get_transform().location
            # update spectator view to follow the vehicle
            update_spectator(spectator, loc)

        print("Starting data collection...")
        while frame_number < max_frames:
            snapshot = world.tick()
            loc = vehicle.get_transform().location
            # update spectator view to follow the vehicle
            update_spectator(spectator, loc)

            # wait for all sensor data to be received
            for event in sensor_events.values():
                event.wait()

            # save data to disk
            collect_and_save_data(frame_number, vehicle, sensor_data, cameras, images_dir, ego_poses_dir, lidar_dir, lidar_transform, concat_dir)
            frame_number += 1

            # clear sensor events
            for event in sensor_events.values():
                event.clear()

    except KeyboardInterrupt:
        print('Data collection interrupted by user.')
    except Exception as e:
        print(f'An error occurred: {e}')
    finally:
        # destroy actors
        print('Destroying actors...')
        if original_settings:
            world.apply_settings(original_settings)
        for actor in actor_list:
            if actor is not None:
                actor.destroy()
        print('All actors destroyed.')
        print('Done.')

        # update data collection info with end time and total frames
        data_info['end_time'] = datetime.now().isoformat()
        data_info['total_frames'] = frame_number
        data_info_filename = os.path.join(output_dir, 'carla_data_info.json')
        with open(data_info_filename, 'w') as json_file:
            json.dump(data_info, json_file, indent=4)
        print(f'Data collection info saved to {data_info_filename}')

def sensor_callback(image, cam_id, sensor_data, sensor_events):
    """Callback function for camera sensors."""
    sensor_data[cam_id] = image
    sensor_events[cam_id].set()

def lidar_callback(point_cloud, sensor_data, sensor_events):
    """Callback function for LiDAR sensor."""
    sensor_data['lidar'] = point_cloud
    sensor_events['lidar'].set()

def collect_and_save_data(frame_number, vehicle, sensor_data, cameras, images_dir, ego_poses_dir, lidar_dir, lidar_transform, concat_dir):
    """Collects data from sensors and saves it to disk."""
    # Save ego-to-world transformation matrix adjusted for Waymo coordinate system
    vehicle_transform = vehicle.get_transform()
    vehicle_matrix = vehicle_transform_to_waymo_matrix(vehicle_transform)
    ego_pose_filename = os.path.join(ego_poses_dir, f'{frame_number:03d}.txt')
    np.savetxt(ego_pose_filename, vehicle_matrix)
    print(f'Saved ego pose for frame {frame_number}.')

    # Collect and save images
    for camera in cameras:
        cam_id = camera.my_id
        image = sensor_data[cam_id]
        if image is not None:
            image_filename = os.path.join(images_dir, f'{frame_number:03d}_{cam_id}.jpg')
            image.save_to_disk(image_filename)
            print(f'Saved image from camera {cam_id} for frame {frame_number}.')
        else:
            print(f'Warning: No image received from camera {cam_id} for frame {frame_number}.')

    # After saving individual images, concatenate images from specified cameras
    concat_order_ids = [3, 1, 0, 2, 4]  # left, front_left, front, front_right, right
    images_list = []

    for cam_id in concat_order_ids:
        image = sensor_data[cam_id]
        if image is not None:
            # Convert the image to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            # Extract RGB channels
            array = array[:, :, :3]
            # Convert to PIL Image
            pil_image = Image.fromarray(array)
            images_list.append(pil_image)
        else:
            print(f'Warning: No image received from camera {cam_id} for frame {frame_number}, cannot include in concatenated image.')

    if len(images_list) == len(concat_order_ids):
        # All images are available, proceed to concatenate
        total_width = sum([im.width for im in images_list])
        max_height = max([im.height for im in images_list])
        # Create a new blank image with the total width and max height
        concatenated_image = Image.new('RGB', (total_width, max_height))
        # Paste images one by one
        x_offset = 0
        for im in images_list:
            concatenated_image.paste(im, (x_offset, 0))
            x_offset += im.width
        # Save concatenated image to 'concat' folder
        concat_filename = os.path.join(concat_dir, f'{frame_number:03d}.png')
        concatenated_image.save(concat_filename)
        print(f'Saved concatenated image for frame {frame_number}.')
    else:
        print(f'Warning: Not all images are available for concatenation at frame {frame_number}.')

    # Collect and save LiDAR data
    point_cloud = sensor_data['lidar']
    if point_cloud is not None:
        # LiDAR points are in the sensor's local frame
        points = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        xyz = points[:, :3]
        intensity = points[:, 3]

        # Transform LiDAR points to vehicle coordinate system
        lidar_points_vehicle = transform_points(xyz, lidar_transform)

        # Convert to Waymo coordinate system by inverting Y-axis
        lidar_points_vehicle[:, 1] *= -1

        # Origins are the LiDAR sensor location in vehicle coordinates (constant)
        lidar_sensor_location = np.array([lidar_transform.location.x, lidar_transform.location.y, lidar_transform.location.z])
        # Convert sensor location to Waymo coordinate system
        lidar_sensor_location[1] *= -1

        origins = np.tile(lidar_sensor_location, (lidar_points_vehicle.shape[0], 1))

        # Combine origins, points, and intensities
        lidar_data = np.hstack((origins, lidar_points_vehicle, intensity[:, np.newaxis]))
        lidar_filename = os.path.join(lidar_dir, f'{frame_number:03d}.bin')
        lidar_data.astype(np.float32).tofile(lidar_filename)
        print(f'Saved LiDAR data for frame {frame_number}.')
    else:
        print(f'Warning: No LiDAR data received for frame {frame_number}.')

def update_spectator(spectator, loc):
    """Updates the spectator's transform to follow the data collection vehicle from above."""
    spectator.set_transform(carla.Transform(carla.Location(x=loc.x, y=loc.y, z=35), carla.Rotation(yaw=0, pitch=-90, roll=0)))

def vehicle_transform_to_waymo_matrix(transform):
    """Converts vehicle transform to Waymo coordinate system."""
    rotation = transform.rotation
    location = transform.location

    # Convert degrees to radians
    pitch = np.deg2rad(rotation.pitch)
    yaw = np.deg2rad(rotation.yaw)
    roll = np.deg2rad(rotation.roll)

    # Compute rotation matrix components
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    # Rotation matrix in CARLA coordinate system
    R_carla = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]
    ])

    # Adjust for Waymo coordinate system by flipping Y-axis
    T_flip = np.diag([1, -1, 1])
    R_waymo = T_flip @ R_carla @ T_flip

    # Adjust the translation vector
    t = np.array([location.x, -location.y, location.z])  # Flip Y-coordinate

    # Construct 4x4 transformation matrix
    T = np.identity(4)
    T[0:3, 0:3] = R_waymo
    T[0:3, 3] = t

    return T

def transform_to_waymo_dict(transform):
    """Converts transform to Waymo coordinate system and returns as dict."""
    location = transform.location
    rotation = transform.rotation

    # Adjust location
    location_waymo = {
        'x': location.x,
        'y': -location.y,  # Flip Y-coordinate
        'z': location.z
    }

    # Adjust rotation (invert Yaw angle)
    rotation_waymo = {
        'pitch': rotation.pitch,
        'yaw': -rotation.yaw,  # Flip Yaw angle
        'roll': rotation.roll
    }

    return {
        'location': location_waymo,
        'rotation': rotation_waymo
    }

def transform_points(points, transform):
    """Transforms points from sensor coordinate system to vehicle coordinate system."""
    # Rotation matrix
    rotation = transform.rotation
    pitch = np.deg2rad(rotation.pitch)
    yaw = np.deg2rad(rotation.yaw)
    roll = np.deg2rad(rotation.roll)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]
    ])
    # Translation vector
    t = np.array([transform.location.x, transform.location.y, transform.location.z])
    # Apply transformation
    points_transformed = np.dot(R, points.T).T + t
    return points_transformed

if __name__ == '__main__':
    main()
