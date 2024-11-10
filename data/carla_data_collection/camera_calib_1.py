import carla
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='output', help='Directory to save calibration files')
    args = parser.parse_args()

    output_dir = args.output
    intrinsics_dir = os.path.join(output_dir, 'intrinsics')
    extrinsics_dir = os.path.join(output_dir, 'extrinsics')
    os.makedirs(intrinsics_dir, exist_ok=True)
    os.makedirs(extrinsics_dir, exist_ok=True)

    # Camera parameters (should match those used in the data collection script)
    image_width = 1920
    image_height = 1080
    fov = 90  # degrees

    # Compute focal lengths
    f_u = (image_width / 2) / np.tan(np.radians(fov / 2))
    f_v = f_u  # Assuming square pixels
    c_u = image_width / 2
    c_v = image_height / 2

    # Distortion coefficients (assuming zero distortion)
    k1 = k2 = p1 = p2 = k3 = 0.0
    intrinsics = [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]

    # Camera extrinsics (should match the camera transforms in the data collection script)
    camera_transforms = [
        {'id': 0, 'transform': carla.Transform(carla.Location(x=1.5, y=0.0, z=2.4),
                                               carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))},
        {'id': 1, 'transform': carla.Transform(carla.Location(x=1.5, y=-0.5, z=2.4),
                                               carla.Rotation(pitch=0.0, yaw=-45.0, roll=0.0))},
        {'id': 2, 'transform': carla.Transform(carla.Location(x=1.5, y=0.5, z=2.4),
                                               carla.Rotation(pitch=0.0, yaw=45.0, roll=0.0))},
        {'id': 3, 'transform': carla.Transform(carla.Location(x=0.0, y=-0.9, z=2.4),
                                               carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0))},
        {'id': 4, 'transform': carla.Transform(carla.Location(x=0.0, y=0.9, z=2.4),
                                               carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0))},
    ]

    for cam in camera_transforms:
        cam_id = cam['id']
        # Save intrinsic parameters
        intrinsics_filename = os.path.join(intrinsics_dir, f'{cam_id}.txt')
        np.savetxt(intrinsics_filename, intrinsics)
        print(f'Saved intrinsics for camera {cam_id}.')

        # Compute and save extrinsic matrix adjusted for Waymo coordinate system
        transform = cam['transform']
        extrinsic_matrix = transform_to_waymo_matrix(transform)
        extrinsics_filename = os.path.join(extrinsics_dir, f'{cam_id}.txt')
        np.savetxt(extrinsics_filename, extrinsic_matrix)
        print(f'Saved extrinsics for camera {cam_id}.')

    print('Camera intrinsics and extrinsics have been saved.')

def transform_to_waymo_matrix(transform):
    # Adjust the transform to Waymo coordinate system
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

    # Rotation matrix (from camera to vehicle coordinate system in CARLA)
    R_carla = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]
    ])

    # Adjust for Waymo coordinate system by flipping the Y-axis
    T_flip = np.diag([1, -1, 1])
    R_waymo = T_flip @ R_carla @ T_flip

    # Adjust the translation vector
    t = np.array([location.x, -location.y, location.z])  # Flip Y-coordinate

    # Construct 4x4 extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[0:3, 0:3] = R_waymo
    extrinsic_matrix[0:3, 3] = t

    return extrinsic_matrix

if __name__ == '__main__':
    main()
