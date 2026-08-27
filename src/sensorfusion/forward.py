"""
Forward propogation for sensor fusion model. 
IMU integration
"""
#IMU used is BNO055, Lidar is RPLidar A1M8, Camera is PiCamZero
from dataclasses import dataclass
from utils.so3_rotation import exp, skew
import numpy as np
from utils.projections import project_points_world
# Per-point logging is extremely expensive on a Raspberry Pi. Enable only when
# diagnosing a specific scan.
DEBUG_LIDAR = False
#Kalman filter --- Prediction, Update/Correction

@dataclass
class State:
    R:np.ndarray  #Orientation (3,3)
    p:np.ndarray  #Translation (3,)
    v:np.ndarray  #Velocity (3,)
    bg:np.ndarray #Gyro Bias (3,)
    ba:np.ndarray #Accel Bias (3,)
    g:np.ndarray #Gravity (3,)

class ESIKFStateEstimator:
    def __init__(self):
        self.P = 1 * np.eye(18) # process covariance matrix
        self.Q = np.eye(18) # process noise covariance
        self.R = np.eye(3) # measurement matrix
        dt = 0.01  # IMU is at 100Hz, so time step is 0.01 seconds
        self.state = State(
            R = np.eye(3,3),
            p = np.zeros(3),
            v = np.zeros(3),
            bg = np.zeros(3),
            ba = np.zeros(3),
            g = np.array([0,0,-9.81]),
        )
        # Runtime diagnostics consumed by the LiDAR worker.
        self.last_lidar_update_applied = False
        self.last_lidar_residual_count = 0
        self.last_lidar_residual_norm = None
    def compute_jacobian(self, x_prev, u, dt):
        #Computing Jacobian of the state model wrt the state error delta_x
        # dt = dt
        A = np.eye(18)
        a_I = u[1] - x_prev.ba
        w_I = u[0] - x_prev.bg
        a_skew = skew(a_I)
        w_skew = skew(w_I)

        # 1. Rotation error evolution
        # Approximation of expm(-w_skew * dt)
        A[0:3, 0:3] = np.eye(3) - (w_skew * dt) 
        
        # 2. Gyro bias impact on rotation error
        A[0:3, 9:12] = -np.eye(3) * dt

        A[3:6, 0:3] = -0.5 * x_prev.R @ a_skew * dt**2
        A[3:6, 6:9] = np.eye(3) * dt
        A[3:6, 12:15] = -0.5 * x_prev.R * dt**2
        A[3:6, 15:18] = np.eye(3) * 0.5 * dt**2

        A[6:9, 0:3] = -x_prev.R @ a_skew * dt
        A[6:9, 12:15] = -x_prev.R * dt
        A[6:9, 15:18] = np.eye(3) * dt

        return A
    
    def predict(self, u: list, dt):#u = [w,a]
        x_prev = self.state

        A = self.compute_jacobian(x_prev, u, dt)#compute the jacobian of the state transition model

        # Implement the prediction step of the Kalman filter here
        ang_act = u[0] - x_prev.bg #wm = wa + bg + ng -> wa = wm - bg - ng
        accel = (x_prev.R @ (u[1] - x_prev.ba) - x_prev.g)

        delta_theta = ang_act * dt
        delta_R = exp(delta_theta)
        self.state.R = self.state.R @ delta_R  #del_theta = w*del_t --> converted to proper SO(3) before adding to the rotation matrix(SO(3))
        self.state.p += (self.state.v * dt) + (0.5 * accel * dt * dt) 
        self.state.v += accel * dt

        #Covariance update
        self.P = A @ self.P @ A.T + self.Q
        return self.state, self.P

    def lidar_update(self, scan, state, lidar_points_compensated, map):
        """
        LiDAR based update.
        When the LiDAR scan is motion compensated, do the residual computation 
        and update the state using the ESIKF filter.
        """
        #first transform the LiDAR points to the world frame using the current state
        lidar_imu_extrinsic = np.eye(4) #assuming the LiDAR and IMU are co-located
        global_imu_extrinsic = np.eye(4) #assuming the IMU is at the origin of the world frame (to transform imu to global)
        camera_imu_extrinsic = np.eye(4)
        T_GI = np.eye(4) #homogeneous transformation matrix built from current ESIKF state at the scan end time t_k.

        # for point in lidar_points_compensated:
        #     # Transform each point from lidar frame to the world frame based on current pose
        #     point_world = T_GI @ lidar_imu_extrinsic @ np.append(point, 1)
        #     points_world.append(point_world[:3])  # Extract the 3D coordinates

        #Compute the residuals between each point and the nearest plane in the world map
        total_res = 0
        self.last_lidar_update_applied = False
        self.last_lidar_residual_count = 0
        self.last_lidar_residual_norm = None
        # state_updated = np.array()
        eps = 0.01
        MIN_INITIAL_POINTS = 30

        #Iterated Kalman Update
        if scan is not None:
            #Skip update for the initial scan
            if map.num_points() < MIN_INITIAL_POINTS:
                #add lidar points directly to the map for initial scans
                points_world = project_points_world(lidar_points_compensated,
                                                    state, lidar_imu_extrinsic)
                map.add_points(points_world)
                if DEBUG_LIDAR:
                    print("Not enough points in the map")
                return

            if map.is_empty():
                points_world = []
                T_GI[:3, :3] = state.R
                T_GI[:3, 3] = state.p
                # for point_lidar in lidar_points_compensated:
                #     point = T_GI @ lidar_imu_extrinsic @ np.append(point_lidar, 1)
                #     points_world.append(point[:3])
                points_world = (T_GI @ lidar_imu_extrinsic @ lidar_points_compensated.T).T
                map.add_points(points_world)
                
                # Clean up IMU buffer and skip EKF update
                # imu_measurement_buffer = [m for m in imu_measurement_buffer if m[0] >= lidar_prev_scan_time]
                return # Skips to the camera logic

            kalman_gain = None
            H = None
            max_iterations = 10
            for iter_count in range(max_iterations):
                points_world = []
                H_list = []
                residuals = []
                T_GI[:3, :3] = state.R
                T_GI[:3, 3] = state.p

                for point_lidar in lidar_points_compensated:

                    # Transform each point from lidar frame to the world frame based on current pose
                    point = T_GI @ lidar_imu_extrinsic @ np.append(point_lidar, 1)
                    points_world.append(point[:3])  # Extract the 3D coordinates
                    point = point[:3]

                    #find the k nearest points from the global map and fit a plane
                    # 4. Fit a plane using SVD
                    neighbors = map.query(point)
                    if neighbors is None or (len(neighbors) < 3):
                        continue
                    if DEBUG_LIDAR:
                        print("Number of neighbors for a point", len(neighbors))
                    center = np.mean(neighbors, axis=0) 
                    centered_neighbors = neighbors - center
                    #find the normal to the plane based on the SVD
                    _, s, vh = np.linalg.svd(centered_neighbors)

                    if s[0] < 1e-6:
                        continue  # degenerate, no structure
                    ratio = s[1] / s[0]
                    if ratio > 0.3:
                        normal = vh[-1, :]  # Plane normal vector
                        if DEBUG_LIDAR:
                            print("Singular Values for plane:", s)
                        #find the residual based on the normal and the center point
                        vec = point - center
                        res = float(np.dot(normal, vec))
                    elif s[1] / s[0] < 0.15:  # optional stricter check, or just an else
                        # Edge/line feature: point-to-line residual instead of discarding
                        direction = vh[0, :]  # principal direction of the line
                        vec = point - center
                        perp = vec - np.dot(vec, direction) * direction  # component perpendicular to the line
                        res = float(np.linalg.norm(perp))
                        normal = perp / (res + 1e-9)  # "normal" here is the residual direction for the Jacobian
                    else:
                        continue  # ambiguous, skip
                    #reject large residuals
                    if abs(res) > 0.20:
                        continue
                    if DEBUG_LIDAR:
                        print("Plane residual:", res)
                    residuals.append(res)
                    #lidar jacobian computation
                    H_pos = normal.T
                    H_rot = -normal @ state.R @ skew(point_lidar)
                    H_k = np.hstack([
                        H_rot,
                        H_pos,
                        np.zeros(3),  # velocity
                        np.zeros(3),  # gyro bias
                        np.zeros(3),  # accel bias
                        np.zeros(3),  # gravity  
                    ])      
                    H_list.append(H_k)

                if len(H_list) == 0:
                    if DEBUG_LIDAR:
                        print("No valid LiDAR points for EKF update")
                    break

                if DEBUG_LIDAR:
                    print("Len of H_list", len(H_list))

                H = np.vstack(H_list)
                r = -np.asarray(residuals)
                self.last_lidar_residual_count = len(r)
                self.last_lidar_residual_norm = float(np.linalg.norm(r))

                if DEBUG_LIDAR:
                    print("Residual norm", np.linalg.norm(r))

                # error_pred = self.P @ H.T
                sigma_lidar = 0.02
                # lidar_sensor_noise = sigma_lidar**2 * np.eye(len(r)) #error in the lidar measurement
                # error_meas = H @ cov @ H.T + lidar_sensor_noise
                # kalman_gain = error_pred @ np.linalg.inv(error_meas) #calculating the Kalman Gain

                R_inv = (1.0 / sigma_lidar**2) * np.eye(len(r)) 
                P_inv = np.linalg.inv(self.P)
                kalman_gain = np.linalg.inv(H.T @ R_inv @ H + P_inv) @ (H.T @ R_inv)

                dx = kalman_gain @ r #error-state vector

                if DEBUG_LIDAR:
                    print("State correction error norm", np.linalg.norm(dx))
                if np.linalg.norm(dx) < eps:
                    break

                theta_rot = dx[0:3]
                state.R = state.R @ exp(theta_rot)
                state.p  += dx[3:6]
                state.v  += dx[6:9]
                state.bg += dx[9:12]
                state.ba += dx[12:15]
                state.g  += dx[15:18]

            #Prevent crash when there is no LiDAR update
            if kalman_gain is not None and H is not None:
                I = np.eye(self.P.shape[0])
                self.P = (I - kalman_gain @ H) @ self.P #covariance update
                self.last_lidar_update_applied = True

            return points_world
