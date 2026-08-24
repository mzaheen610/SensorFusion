"""
Helper utils for 3D - Camera projections and visual calculations
"""
import numpy as np

#Camera intrinsic parameters - focal length and center 
#based on theoretical values for 640x480 resoulution
focals = [529.6, 528.8] #unit is pixel
center = [320, 240]

def project_points_to_frame(points, cam_imu_transform, glob_imu):
    #project the visual map points to the current image frame
    pixels = []
    glob_imu_inv = np.linalg.inv(glob_imu)
    for point in points:
        camera_coords = cam_imu_transform @ glob_imu_inv @ point
        pixel_coords = project(camera_coords)
        pixels.append(pixel_coords)

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

def calculate_photometric_error(curr_frame, ref_frame, pixels):
    pass
def visual_update(frame, state):
    #compute the visual update based on the camera scan
    pass