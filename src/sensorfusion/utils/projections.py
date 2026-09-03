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
        #project the points first to the camera coordinates
        camera_coords = cam_imu_transform @ glob_imu_inv @ point
        #project points to the image frame
        pixel_coords = project(camera_coords)
        if (pixel_coords[0]< 640) and (pixel_coords[0]< 480):
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

def calculate_photometric_error(curr_frame, ref_frame, pixels):
    m = len(curr_frame)
    n = len(curr_frame[0])
    residual = []
    #compute the photometric residual for each pixel
    for i in range(m):
        for j in range(n):
            res = curr_frame[i][j] - ref_frame[i][j]
            residual.append(res)
    return residual

def visual_update(frame, state):
    #compute the visual update based on the camera scan
    pass