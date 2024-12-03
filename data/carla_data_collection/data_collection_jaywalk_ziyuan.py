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
        traffic_manager.vehicle_percentage_speed_difference(vehicle, 30)  # Ego vehicle goes 10% slower than speed limit
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

      
        ego_v_spawn_point = spawn_point
        # walker_spawn_loc = ego_v_spawn_point
        # walker_spawn_loc.location.x += 8
        # walker_spawn_loc.location.z += 1
        # walker_spawn_loc.location.y += 0
        
        walker, walker_spawn_trans = spawn_pedestrian(ego_v_spawn_point, blueprint_library, world)
        # walker = spawn_walker(world, blueprint_library, vehicle, world.get_map())
        if walker:
            actor_list.append(walker)  # add walker to actor list
            print("extended walker and controller")
            # Compute target location
            target_location = carla.Location(
                x=walker_spawn_trans.location.x,
                y=walker_spawn_trans.location.y - 15,  # Move along negative Y-axis
                z=walker_spawn_trans.location.z
            )
            # waypoint = world.get_map().get_waypoint(target_location)
            # if waypoint is None:
            #     print("Target location is not on the navigation mesh.")
            #     return
            # waypoint = world.get_map().get_waypoint(target_location)
            # if waypoint is None:
            #     print("Target location is not on the navigation mesh.")
            #     return
            world.debug.draw_point(target_location, size=0.2, color=carla.Color(0, 0, 255), life_time=10)

            # Set up the walker control
            walker_control = carla.WalkerControl()
            direction = target_location - walker_spawn_trans.location  # Compute direction vector
            walker_control.direction = direction.make_unit_vector()  # Normalize the direction vector
            walker_control.speed = 1.8  # Walking speed in m/s
            walker_control.jump = False  # Disable jump

            # setup_jaywalking_behavior(walker_spawn_trans, walker, world, walker_control)
        else:
            print("Failed to spawn walker or controller. Exiting data collection.")
            return


        wait_data_collection = 3.2  # seconds
        # let the vehicles move for {wait for data} seconds before data collection starts
        print(f"wait Vehicles are moving for {wait_data_collection} seconds before data collection starts...")
        for _ in range(int(wait_data_collection * args.fps)):
            walker.apply_control(walker_control)
            snapshot = world.tick()
            loc = vehicle.get_transform().location

            # update spectator view to follow the vehicle
            update_spectator(spectator, loc)
        
        vehicle.set_autopilot(False, traffic_manager.get_port())
        control = carla.VehicleControl()
        control.throttle = 0.2
        control.brake = 0.0
        control.steer = 0.0
        vehicle.apply_control(control)

        print("Starting data collection...")
        while frame_number < max_frames:
            walker.apply_control(walker_control)
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
        

# def spawn_walker(world, blueprint_library, vehicle, carla_map):
#     # Get ego vehicle's waypoint
#     ego_waypoint = carla_map.get_waypoint(vehicle.get_location(), project_to_road=True)
    
#     walker_trans = ego_waypoint.transform
    
#     walker_trans.location.x += 6
#     walker_trans.location.z += 1
    
#     # Spawn the walker
#     walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
#     walker = world.spawn_actor(walker_bp, walker_trans)
#     print(f"Pedestrian spawned at location: {walker_trans.location}")
#     world.debug.draw_point(walker_trans.location, size=0.2, color=carla.Color(255,0,0), life_time=10)

    
#     return walker
    
    
    
        
def spawn_pedestrian(ego_v_spawn_point, blueprint_library, world):
    """Spawns a pedestrian that will jaywalk in front of the ego vehicle."""
    # Create a new Transform based on ego_v_spawn_point
    walker_spawn_trans = carla.Transform(
        location=carla.Location(
            x=ego_v_spawn_point.location.x + 30,
            y=ego_v_spawn_point.location.y + 8,
            z=ego_v_spawn_point.location.z
        ),
        rotation=carla.Rotation(
            pitch=ego_v_spawn_point.rotation.pitch,
            yaw=ego_v_spawn_point.rotation.yaw - 90,  # Adjust yaw
            roll=ego_v_spawn_point.rotation.roll
        )
    )

    # Spawn the walker
    walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
    walker = world.try_spawn_actor(walker_bp, walker_spawn_trans)
    if walker is None:
        print(f"Failed to spawn walker at {walker_spawn_trans.location}. Is the location valid and unoccupied?")
        return None, None

    print(f"Walker spawned at location: {walker_spawn_trans.location}")

    # Visualize the spawn location
    world.debug.draw_point(walker_spawn_trans.location, size=0.2, color=carla.Color(255, 0, 0), life_time=10)

    return walker, walker_spawn_trans


# def setup_jaywalking_behavior(target_location, walker, control):
#     """Controls the pedestrian to cross the road using carla.WalkerControl."""
#     if not walker:
#         print("Walker is not initialized.")
#         return

#     # Move walker toward the target location
#     print("Walker starts moving to target location.")
    
#     walker.apply_control(control)
#     # world.tick()
#     # walker_location = walker.get_location()
#     # distance_to_target = walker_location.distance(target_location)
#         # print(f"Walker current location: {walker_location}, Distance to target: {distance_to_target:.2f}")
#         # if distance_to_target < 0.5:  # Stop when walker is close to the target
#         #     print("Walker reached the target location.")
#         #     break

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

    # # After saving individual images, concatenate images from specified cameras
    # concat_order_ids = [3, 1, 0, 2, 4]  # left, front_left, front, front_right, right
    # images_list = []

    # for cam_id in concat_order_ids:
    #     image = sensor_data[cam_id]
    #     if image is not None:
    #         # Convert the image to numpy array
    #         array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    #         array = np.reshape(array, (image.height, image.width, 4))
    #         # Extract RGB channels
    #         array = array[:, :, :3]
    #         # Convert to PIL Image
    #         pil_image = Image.fromarray(array)
    #         images_list.append(pil_image)
    #     else:
    #         print(f'Warning: No image received from camera {cam_id} for frame {frame_number}, cannot include in concatenated image.')

    # if len(images_list) == len(concat_order_ids):
    #     # All images are available, proceed to concatenate
    #     total_width = sum([im.width for im in images_list])
    #     max_height = max([im.height for im in images_list])
    #     # Create a new blank image with the total width and max height
    #     concatenated_image = Image.new('RGB', (total_width, max_height))
    #     # Paste images one by one
    #     x_offset = 0
    #     for im in images_list:
    #         concatenated_image.paste(im, (x_offset, 0))
    #         x_offset += im.width
    #     # Save concatenated image to 'concat' folder
    #     concat_filename = os.path.join(concat_dir, f'{frame_number:03d}.png')
    #     concatenated_image.save(concat_filename)
    #     print(f'Saved concatenated image for frame {frame_number}.')
    # else:
    #     print(f'Warning: Not all images are available for concatenation at frame {frame_number}.')

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
