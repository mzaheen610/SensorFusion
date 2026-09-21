"""
Helper utils for 3D - Camera projections and visual calculations
"""
import os
import numpy as np

# Camera intrinsic parameters - loaded from calibration file if available,
# otherwise fallback to theoretical 640x480 values.
calib_paths = [
    os.path.join(os.path.dirname(__file__), "..", "camera", "calibration_params.npz"),
    os.path.join(os.path.dirname(__file__), "..", "calibration_params.npz"),
    "calibration_params.npz",
    "src/sensorfusion/camera/calibration_params.npz",
]

calib_file = next((p for p in calib_paths if os.path.exists(p)), None)
dist_coeffs = None
K = None

if calib_file is not None:
    try:
        calib_data = np.load(calib_file)
        K = calib_data["mtx"]
        dist_coeffs = calib_data["dist"]
        focals = [float(K[0, 0]), float(K[1, 1])]
        center = [float(K[0, 2]), float(K[1, 2])]
    except Exception:
        focals = [529.6, 528.8]
        center = [320.0, 240.0]
else:
    focals = [529.6, 528.8]
    center = [320.0, 240.0]


def project_points_to_frame(points, cam_imu_transform, glob_imu):
    #project the visual map points to the current image frame
    pixels = []
    glob_imu_inv = np.linalg.inv(glob_imu)
    for point in points:
        #project the points first to the camera coordinates
        point_h = np.append(point, 1.0)
        camera_coords = cam_imu_transform @ glob_imu_inv @ point_h
        if not np.all(np.isfinite(camera_coords)) or camera_coords[2] <= 0:
            continue
        #project points to the image frame
        pixel_coords = project(camera_coords)
        if (0 <= pixel_coords[0] < 640
                and 0 <= pixel_coords[1] < 480):
            #reject points outside the frame range (640, 480)
            pixels.append((point, pixel_coords))
    return pixels

def project_points_world(points, state, lidar_imu_extrinsic):
    T_GI = np.eye(4)
    points_world = []
    T_GI[:3, :3] = state.R
    T_GI[:3, 3] = state.p
    for point_lidar in points:
        point = T_GI @ lidar_imu_extrinsic @ np.append(point_lidar, 1)
        points_world.append(point[:3])
    return points_world

def project(coords):
    #3D to pinhole camera projection
    u = focals[0]*coords[0]/coords[2] + center[0]
    v = focals[1]*coords[1]/coords[2] + center[1]
    return (u,v) #the pixel coord equivalent of the 3D points

def calculate_photometric_error(curr_frame, ref_frame, pixels=None):
    curr = np.asarray(curr_frame, dtype=np.float32)
    ref = np.asarray(ref_frame, dtype=np.float32)
    residual = (curr - ref).ravel().tolist()
    return residual
