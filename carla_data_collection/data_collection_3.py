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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=25, help='Data collection frequency (frames per second)')
    parser.add_argument('--output', type=str, default='carla_data', help='Directory to save output data')
    parser.add_argument('--max_f', type=int, default=50, help='Number of frames to collect')
    args = parser.parse_args()

    output_dir = args.output

    # Create a dictionary to store data collection info, no per-frame detailed data
    data_info = {
        'start_time': datetime.now().isoformat(),
        'fps': args.fps,
        'sensors': [],
        'output_dir': os.path.abspath(output_dir)
    }

    # Connect to CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # seconds
    world = client.get_world()

    # Get blueprint library
    blueprint_library = world.get_blueprint_library()

    # Spawn vehicle
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    i = np.random.randint(0, len(spawn_points))
    print(f"Spawning vehicle at spawn point {i}.")
    spawn_point = spawn_points[i]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Set up vehicle autopilot
    vehicle.set_autopilot(True)

    # Define camera and LiDAR blueprints
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

    # Set camera attributes
    image_width = 1920
    image_height = 1080
    fov = 90  # degrees
    camera_bp.set_attribute('image_size_x', str(image_width))
    camera_bp.set_attribute('image_size_y', str(image_height))
    camera_bp.set_attribute('fov', str(fov))

    # Define five different camera perspectives
    camera_transforms = [
        {'id': 0, 'transform': carla.Transform(carla.Location(x=1.5, y=0.0, z=2.4),
                                               carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))},  # FRONT
        {'id': 1, 'transform': carla.Transform(carla.Location(x=1.5, y=-0.5, z=2.4),
                                               carla.Rotation(pitch=0.0, yaw=-45.0, roll=0.0))},  # FRONT_LEFT
        {'id': 2, 'transform': carla.Transform(carla.Location(x=1.5, y=0.5, z=2.4),
                                               carla.Rotation(pitch=0.0, yaw=45.0, roll=0.0))},  # FRONT_RIGHT
        {'id': 3, 'transform': carla.Transform(carla.Location(x=0.0, y=-0.9, z=2.4),
                                               carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0))},  # SIDE_LEFT
        {'id': 4, 'transform': carla.Transform(carla.Location(x=0.0, y=0.9, z=2.4),
                                               carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0))},  # SIDE_RIGHT
    ]

    # Add camera sensor info to data_info
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

    # Create shared data structures for sensors
    sensor_data = {}
    sensor_events = {}
    cameras = []
    camera_0 = None  # Placeholder to save reference to camera 0

    # Create cameras and attach to vehicle
    for cam in camera_transforms:
        camera = world.spawn_actor(camera_bp, cam['transform'], attach_to=vehicle)
        camera.my_id = cam['id']  # Use custom attribute name
        sensor_data[camera.my_id] = None
        sensor_events[camera.my_id] = threading.Event()
        camera.listen(lambda image, cam_id=camera.my_id: sensor_callback(image, cam_id, sensor_data, sensor_events))
        cameras.append(camera)
        if camera.my_id == 0:
            camera_0 = camera  # Save reference to camera 0

    # Set LiDAR attributes and attach to vehicle
    lidar_bp.set_attribute('range', '100.0')
    lidar_bp.set_attribute('rotation_frequency', str(args.fps))
    lidar_bp.set_attribute('points_per_second', '1000000')
    lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.5))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    sensor_data['lidar'] = None
    sensor_events['lidar'] = threading.Event()
    lidar.listen(lambda point_cloud: lidar_callback(point_cloud, sensor_data, sensor_events))

    # Add LiDAR sensor info to data_info
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

    # Create data storage directories
    images_dir = os.path.join(output_dir, 'images')
    ego_poses_dir = os.path.join(output_dir, 'ego_pose')
    lidar_dir = os.path.join(output_dir, 'lidar')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ego_poses_dir, exist_ok=True)
    os.makedirs(lidar_dir, exist_ok=True)

    # Get spectator and set initial transform
    spectator = world.get_spectator()
    # Set spectator view to camera 0
    if camera_0 is not None:
        spectator.set_transform(camera_0.get_transform())

    # Set synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    frame_number = 0
    max_frames = args.max_f  # For example, collect 50 frames

    # Let the vehicle move for 5 seconds before starting data collection
    print("Vehicle is moving for 5 seconds before data collection starts...")
    for _ in range(int(5 * args.fps)):
        snapshot = world.tick()
        # Update spectator to follow camera 0
        update_spectator(spectator, camera_0)

    print("Starting data collection...")
    try:
        while frame_number < max_frames:
            snapshot = world.tick()
            # Update spectator to follow camera 0
            update_spectator(spectator, camera_0)

            # Wait for all sensor data to be available
            for event in sensor_events.values():
                event.wait()

            # Collect and save data
            collect_and_save_data(frame_number, vehicle, sensor_data, cameras, images_dir, ego_poses_dir, lidar_dir, lidar_transform)
            frame_number += 1

            # Clear events for next frame
            for event in sensor_events.values():
                event.clear()

    except KeyboardInterrupt:
        print('Data collection interrupted by user.')

    finally:
        # Destroy actors
        print('Destroying actors...')
        for camera in cameras:
            camera.stop()
            camera.destroy()
        lidar.stop()
        lidar.destroy()
        vehicle.destroy()
        world.apply_settings(original_settings)
        print('Done.')

        # Update data_info after data collection and save as JSON file
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

def collect_and_save_data(frame_number, vehicle, sensor_data, cameras, images_dir, ego_poses_dir, lidar_dir, lidar_transform):
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

def update_spectator(spectator, camera):
    # Update the spectator's transform to match the camera's transform
    if spectator is not None and camera is not None:
        camera_transform = camera.get_transform()
        spectator.set_transform(camera_transform)

def vehicle_transform_to_waymo_matrix(transform):
    """Convert vehicle transform to Waymo coordinate system."""
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
    """Convert transform to Waymo coordinate system and return as dict."""
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
    """Transform points from sensor coordinate system to vehicle coordinate system."""
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
