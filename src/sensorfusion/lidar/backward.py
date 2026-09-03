"""
Propogate the Lidar scans backward using the IMU poses
to correct motion distortion
"""

from math import pi
from utils.so3_rotation import exp
import numpy as np
from copy import deepcopy

def compute_prev_pose(current_state, delta_time, gyro, accel):
    x_prev = deepcopy(current_state)
    # A = self.compute_jacobian(x_prev, u)#compute the jacobian of the state transition model

    # Implement the prediction step of the Kalman filter here
    ang_act = gyro - x_prev.bg  #wm = wa + bg + ng -> wa = wm - bg - ng
    accel = (x_prev.R @ (accel - x_prev.ba) - x_prev.g)

    delta_theta = ang_act * delta_time
    delta_R = exp(delta_theta)
    x_prev.R = x_prev.R @ delta_R.T  #del_theta = w*del_t --> converted to proper SO(3) before adding to the rotation matrix(SO(3))
    x_prev.p -= (x_prev.v * delta_time) - (0.5 * accel * delta_time**2)
    x_prev.v -= accel * delta_time

    #Covariance update
    # self.P = A @ self.P @ A.T + self.Q
    return x_prev

def backprop(scan_end_time, prev_scan_time, imu_pose, scan, imu_measurement_buffer):
    """
    Backward propagate LiDAR points using IMU poses to correct motion distortion
    """
    compensated_points = []
    angular_rate_lidar = 2 * pi * 10 # 10Hz LiDAR scan frequency
    for point in scan:
        #Find the delta time between scan end and the point sampled time
        # point_time = point[1] / angular_rate_lidar
        observed_period = scan_end_time - prev_scan_time
        scan_period = np.clip(observed_period, 0.08, 0.15)
        point_time = (point[1] / 360.0) * scan_period
        point_abs_time = prev_scan_time + point_time

        delta_time = scan_end_time - (prev_scan_time + point_time)
        pose_j = deepcopy(imu_pose)
        current_time = scan_end_time
        valid_point = False
        # print("Scan end time: ", scan_end_time)
        # print("Prev scan time: ", prev_scan_time)
        # print("Point time: ", point_time)
        #Backpropogate the state using IMU measurements till the point time is reached
        for imu_time, gyro, accel in reversed(imu_measurement_buffer):
            if imu_time >= current_time:
                continue
            #for points captured after the last IMU measurement, we use the last IMU measurement to backpropogate the state
            if imu_time <= point_abs_time:
                dt = current_time - point_abs_time
                pose_j = compute_prev_pose(pose_j, dt, gyro, accel)
                valid_point = True
                break
            dt = current_time - imu_time
            if dt <= 0.0 or dt > 0.1:
                break
            pose_j = compute_prev_pose(pose_j, dt, gyro, accel)
            current_time = imu_time

        if not valid_point:
            continue
        #Transform the point to the scan end time frame using the pose_j
        angle = np.deg2rad(point[1])
        distance = point[2] / 1000.0 #convert mm to meters
        point_body = np.array([distance * np.cos(angle), distance * np.sin(angle), 0]) #point in body frame
        R_kj = imu_pose.R.T @ pose_j.R
        p_kj = imu_pose.R.T @ (pose_j.p - imu_pose.p)

        # print("Pose at scan end time: ", imu_pose.p)
        # print("Pose at point time: ", pose_j.p)
        # print("R_kj: ", R_kj)
        # print("p_kj: ", p_kj)
        projected_point = R_kj @ point_body + p_kj
        compensated_points.append(projected_point)
    return np.asarray(compensated_points, dtype=float).reshape(-1, 3)