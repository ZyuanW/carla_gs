import numpy as np
from plyfile import PlyData, PlyElement

def npy_to_ply(npy_file, ply_file, include_color=False):
    """
    Converts an .npy file to a .ply point cloud file.

    Parameters:
        npy_file (str): Path to the input .npy file containing point cloud data.
        ply_file (str): Path to the output .ply file.
        include_color (bool): Whether the .npy file includes color data (RGB).
                              If True, the .npy file is expected to have shape (N, 6).
    """
    # Load the .npy file
    data = np.load(npy_file)
    
    if include_color:
        assert data.shape[1] == 6, "Expected shape (N, 6) for color data."
        vertices = [
            tuple(row) for row in data
        ]  # Convert each row to a tuple
        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1")
        ]
    else:
        assert data.shape[1] == 3, "Expected shape (N, 3) for point data."
        vertices = [
            tuple(row) for row in data
        ]
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]

    # Create a structured array
    vertex_array = np.array(vertices, dtype=dtype)

    # Create a PlyElement
    ply_element = PlyElement.describe(vertex_array, "vertex")

    # Write to a .ply file
    PlyData([ply_element]).write(ply_file)
    print(f"Successfully saved {ply_file}")

# Example usage:
# Convert a .npy file with only XYZ coordinates
npy_file = "/home/user/persistent/carla_gs/data/carla/break_test_1/dinov2_vitb14/000_0.npy"
npy_to_ply("example.npy", "output.ply", include_color=False)

# # Convert a .npy file with XYZRGB data
# npy_to_ply("example_with_color.npy", "output_with_color.ply", include_color=True)