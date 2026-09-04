"""
Forward propogation for sensor fusion model. 
IMU integration
"""
#IMU used is BNO055, Lidar is RPLidar A2M12, Camera is PiCamZero
from dataclasses import dataclass
from utils.so3_rotation import exp, skew
import numpy as np
from utils.projections import project_points_world
# Per-point logging is extremely expensive on a Raspberry Pi. Enable only when
# diagnosing a specific scan.
DEBUG_LIDAR = True
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
        self.q_rot = 1e-3      # rad^2/s -- gyro noise density
        self.q_pos = 1e-4      # m^2/s (loosely, position integrates velocity)
        self.q_vel = 1e-2      # (m/s)^2/s -- accel noise density
        self.q_gyro_bias = 1e-8   # rad^2/s -- gyro bias random walk (slow)
        self.q_accel_bias = 1e-6  # (m/s^2)^2/s -- accel bias random walk (slow)
        self.q_gravity = 1e-10    # near-static; only nudge via correlation
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
        self.last_lidar_association_count = 0

    def compute_process_noise(self, dt):
        """Q, scaled by dt 
        """
        Q = np.zeros((18, 18))
        Q[0:3, 0:3]   = self.q_rot * dt * np.eye(3)
        Q[3:6, 3:6]   = self.q_pos * dt * np.eye(3)
        Q[6:9, 6:9]   = self.q_vel * dt * np.eye(3)
        Q[9:12, 9:12] = self.q_gyro_bias * dt * np.eye(3)
        Q[12:15, 12:15] = self.q_accel_bias * dt * np.eye(3)
        Q[15:18, 15:18] = self.q_gravity * dt * np.eye(3)
        return Q
    
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
        A[3:6, 15:18] = -np.eye(3) * 0.5 * dt**2

        A[6:9, 0:3] = -x_prev.R @ a_skew * dt
        A[6:9, 12:15] = -x_prev.R * dt
        A[6:9, 15:18] = -np.eye(3) * dt

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
        self.P = A @ self.P @ A.T + self.compute_process_noise(dt)
        return self.state, self.P

    def lidar_update(self, scan, state, P_copy, lidar_points_compensated, map):
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

        #Compute the residuals between each point and the nearest plane in the world map
        total_res = 0
        self.last_lidar_update_applied = False
        self.last_lidar_residual_count = 0
        self.last_lidar_residual_norm = None
        self.last_lidar_association_count = 0
        # state_updated = np.array()
        eps = 0.01
        MIN_INITIAL_POINTS = 30
        MIN_ASSOCIATIONS = 10
        P_new = P_copy # Default fallback

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
                return None, False, P_copy

            if map.is_empty():
                points_world = []
                T_GI[:3, :3] = state.R
                T_GI[:3, 3] = state.p
                
                # Generate homogeneous coordinates manually to avoid broadcast errors
                lidar_homo = np.hstack([lidar_points_compensated, np.ones((len(lidar_points_compensated), 1))])
                points_world = (T_GI @ lidar_imu_extrinsic @ lidar_homo.T).T[:, :3]
                map.add_points(points_world)
                
                # Clean up IMU buffer and skip EKF update
                # imu_measurement_buffer = [m for m in imu_measurement_buffer if m[0] >= lidar_prev_scan_time]
                return points_world, False, P_copy # Skips to the camera logic

            # --- PRE-COMPUTE DATA ASSOCIATIONS ONCE ---
            valid_associations = []
            T_GI_init = np.eye(4)
            T_GI_init[:3, :3] = state.R
            T_GI_init[:3, 3] = state.p

            for point_lidar in lidar_points_compensated:
                # Transform each point from lidar frame to the world frame based on current pose
                point = T_GI_init @ lidar_imu_extrinsic @ np.append(point_lidar, 1)
                point_world_coords = point[:3]

                #find the k nearest points from the global map and fit a plane
                # 4. Fit a plane using SVD
                neighbors = map.query(point_world_coords)
                if neighbors is None or (len(neighbors) < 3):
                    continue
                # if DEBUG_LIDAR:
                #     print("Number of neighbors for a point", len(neighbors))
                
                center = np.mean(neighbors, axis=0) 
                centered_neighbors = neighbors - center

                #to restrict plane associations for a 2D lidar based points
                z_spread = neighbors[:, 2].max() - neighbors[:, 2].min()
                MIN_Z_SPREAD_FOR_PLANE = 0.10  

                #find the normal to the plane based on the SVD
                _, s, vh = np.linalg.svd(centered_neighbors)

                if s[0] < 1e-6:
                    continue  # degenerate, no structure
                
                ratio21 = s[1] / (s[0] + 1e-9)  # 2nd-largest spread / largest spread
                ratio31 = s[2] / (s[0] + 1e-9)  # smallest spread / largest spread


                if ratio21 < 0.15:  # optional stricter check, or just an else
                    # Edge/line feature: store direction to calculate dynamic residual later
                    direction = vh[0, :]  # principal direction of the line
                    valid_associations.append(('line', point_lidar, center, direction))
                    if DEBUG_LIDAR:
                        print(
                            f"LINE: s={s}, "
                            f"s1/s0={ratio21:.3f}, "
                            f"s2/s0={ratio31:.3f}"
                        )
                elif ratio21 > 0.3 and ratio31<0.1:
                    normal = vh[-1, :]  # Plane normal vector
                    valid_associations.append(('plane', point_lidar, center, normal))
                    if DEBUG_LIDAR:
                        print("Singular Values for plane:", s)
                        print(
                            f"PLANE: s={s}, "
                            f"s1/s0={ratio21:.3f}, "
                            f"s2/s0={ratio31:.3f}"
                        )
                else:
                    if DEBUG_LIDAR:
                        print(
                            f"REJECT: s={s}, "
                            f"s1/s0={ratio21:.3f}, "
                            f"s2/s0={ratio31:.3f}"
                        )
                    continue  # ambiguous, skip

            kalman_gain = None
            H = None
            self.last_lidar_association_count = len(valid_associations)
            max_iterations = 5
            P_inv = np.linalg.inv(P_copy)

            # --- ITERATED EKF UPDATE ---
            for iter_count in range(max_iterations):
                H_list = []
                residuals = []
                T_GI[:3, :3] = state.R
                T_GI[:3, 3] = state.p

                for assoc_type, point_lidar, center, geom_vec in valid_associations:
                    # Transform each point using the continually updated state
                    point = T_GI @ lidar_imu_extrinsic @ np.append(point_lidar, 1)
                    point_coords = point[:3]

                    #find the residual based on the normal and the center point
                    vec = point_coords - center
                    
                    if assoc_type == 'plane':
                        normal = geom_vec
                        res = float(np.dot(normal, vec))
                    else:
                        # Edge/line feature: point-to-line residual instead of discarding
                        direction = geom_vec
                        perp = vec - np.dot(vec, direction) * direction  # component perpendicular to the line
                        res = float(np.linalg.norm(perp))
                        normal = perp / (res + 1e-9)  # "normal" here is the residual direction for the Jacobian

                    #reject large residuals
                    if abs(res) > 0.80:
                        continue
                    # if DEBUG_LIDAR:
                    #     print("Plane residual:", res)
                        
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
                elif len(H_list) < MIN_ASSOCIATIONS:
                    if DEBUG_LIDAR:
                        print(f"Too few associations ({len(H_list)}), skipping update")
                    break

                if DEBUG_LIDAR:
                    print("Len of H_list", len(H_list))

                H = np.vstack(H_list)
                r = -np.asarray(residuals)
                self.last_lidar_residual_count = len(r)
                self.last_lidar_residual_norm = float(np.linalg.norm(r))

                if DEBUG_LIDAR:
                    print("Residual norm", np.linalg.norm(r))

                sigma_lidar = 0.02

                R_inv = (1.0 / sigma_lidar**2) * np.eye(len(r)) 
                kalman_gain = np.linalg.inv(H.T @ R_inv @ H + P_inv) @ (H.T @ R_inv)

                dx = kalman_gain @ r #error-state vector

                if DEBUG_LIDAR:
                    print("State correction error norm", np.linalg.norm(dx))
                    print("dx:", dx)
                    print("rot:", dx[:3])
                    print("pos:", dx[3:6])
                    print("vel:", dx[6:9])
                    print("bg:", dx[9:12])
                    print("ba:", dx[12:15])
                    print("g:", dx[15:18])
                if np.linalg.norm(dx) < eps:
                    break

                MAX_CORRECTION_NORM = 2.0  # to be tuned based on realistic per-update movement of platform
                if not np.all(np.isfinite(dx)) or np.linalg.norm(dx) > MAX_CORRECTION_NORM:
                    if DEBUG_LIDAR:
                        print(f"Rejecting implausible correction, norm={np.linalg.norm(dx)}")
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
                I = np.eye(P_copy.shape[0])
                P_new = (I - kalman_gain @ H) @ P_copy #covariance update
                self.last_lidar_update_applied = True

            # Generate final world points for the map using the converged state
            T_GI[:3, :3] = state.R
            T_GI[:3, 3] = state.p
            lidar_homo = np.hstack([lidar_points_compensated, np.ones((len(lidar_points_compensated), 1))])
            points_world_final = (T_GI @ lidar_imu_extrinsic @ lidar_homo.T).T[:, :3]

            return points_world_final, self.last_lidar_update_applied, P_new

        return None, False, P_copy

    def zupt_update(self, sigma_zupt=0.05):
        """
        Zero-velocity pseudo-measurement update. Call when LiDAR scan similarity
        indicates the platform is stationary. Directly modifies self.state/self.P.
        """
        state = self.state
        H = np.zeros((3, 18))
        H[:, 6:9] = np.eye(3)   # measurement model observes velocity directly

        r = -state.v            # residual = measured(0) - predicted velocity

        R_zupt = (sigma_zupt ** 2) * np.eye(3)
        S = H @ self.P @ H.T + R_zupt
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ r

        theta_rot = dx[0:3]
        state.R = state.R @ exp(theta_rot)
        state.p  += dx[3:6]
        state.v  += dx[6:9]
        state.bg += dx[9:12]
        state.ba += dx[12:15]
        state.g  += dx[15:18]

        I = np.eye(18)
        self.P = (I - K @ H) @ self.P