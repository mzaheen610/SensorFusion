"""
Forward propogation for sensor fusion model. 
IMU integration
"""
#IMU used is BNO055, Lidar is RPLidar A2M12, Camera is PiCamZero
import time
from dataclasses import dataclass
from utils.so3_rotation import exp, skew, reorthonormalize
import numpy as np
from utils.projections import project_points_world
from lidar.update import copy_state, state_error
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
        self.P[12:15, 12:15] = 1e-8 * np.eye(3) # Initial accel bias covariance
        self.q_rot = 1e-3      # rad^2/s -- gyro noise density
        self.q_pos = 1e-5      # m^2/s (loosely, position integrates velocity)
        self.q_vel = 5e-3      # (m/s)^2/s -- accel noise density
        self.q_gyro_bias = 1e-8   # rad^2/s -- gyro bias random walk (slow)
        self.q_accel_bias = 1e-5  # (m/s^2)^2/s -- accel bias random walk (allows ZUPT adaptation)
        self.q_gravity = 0.0      # frozen; gravity removed onboard BNO055
        self.R = np.eye(3) # measurement matrix
        dt = 0.01  # IMU is at 100Hz, so time step is 0.01 seconds
        self.dt = dt
        self.Q = np.zeros((18, 18))
        self.A = np.eye(18)
        self.state = State(
            R=np.eye(3),
            p=np.zeros(3),
            v=np.zeros(3),
            bg=np.zeros(3),
            ba=np.zeros(3),
            g=np.zeros(3)
        )
        # Runtime diagnostics consumed by the LiDAR worker.
        self.is_static = False
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
    
    def compute_jacobian(self, x_prev: State, u: list, dt):
        #Computing Jacobian of the state model wrt the state error delta_x
        # dt = dt
        A = np.eye(18)
        w_skew = skew(u[0] - x_prev.bg)
        a_skew = skew(u[1] - x_prev.ba)

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
        self.state.R = reorthonormalize(self.state.R) # prevent det(R) runaway
        if getattr(self, 'is_static', False):
            # Windowed mean of horizontal acceleration vector to filter out zero-mean motor vibration oscillations.
            # Filtering the vector before taking the norm avoids rectifying high-frequency vibration into false DC acceleration.
            if not hasattr(self, '_static_accel_ema_vec') or not isinstance(self._static_accel_ema_vec, np.ndarray):
                self._static_accel_ema_vec = np.zeros(2)
            ang_norm = float(np.linalg.norm(ang_act))
            self._static_accel_ema_vec = 0.65 * self._static_accel_ema_vec + 0.35 * accel[:2]
            accel_ema_norm = float(np.linalg.norm(self._static_accel_ema_vec))
            self._static_accel_ema = accel_ema_norm

            # Exit static only on sustained turn (|w| > 0.20) or sustained directional horizontal acceleration (|a_xy| > 0.20)
            if ang_norm > 0.20 or accel_ema_norm > 0.20:
                self.is_static = False
                self._static_accel_ema_vec = np.zeros(2)
                self._static_accel_ema = 0.0
            else:
                self.state.v[:] = 0.0
                self.state.p[2] = 0.0

        if not getattr(self, 'is_static', False):
            self.state.p += (self.state.v * dt) + (0.5 * accel * dt * dt) 
            self.state.v += accel * dt
            self.state.p[2] = 0.0  # Planar robot constraint: 2D motion on ground/table
            self.state.v[2] = 0.0  # Zero unobservable vertical velocity

            # Physical rolling friction / velocity decay when not actively accelerating
            # Prevents braking deceleration transients from coasting open-loop indefinitely
            if np.linalg.norm(accel[:2]) < 0.12:
                self.state.v[:2] *= max(0.0, 1.0 - 2.0 * dt)

            # Safety speed limit for physical platform
            MAX_SPEED = 0.8  # m/s
            speed_xy = float(np.linalg.norm(self.state.v[:2]))
            if speed_xy > MAX_SPEED:
                self.state.v[:2] = (self.state.v[:2] / speed_xy) * MAX_SPEED

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
        MIN_ASSOCIATIONS = 3
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

            # Downsample the lidar points to reduce computational load (cap to at most 40 points)
            n_comp = len(lidar_points_compensated)
            stride = max(1, n_comp // 40)
            points_to_associate = lidar_points_compensated[::stride][:40]

            # Adaptive search radius: during dynamic motion, expand search radius to 2 voxels (±1.0m)
            # to associate fast hand strokes (0.5-1.0m). When static, restrict strictly to 1 voxel (±0.5m) to avoid distant false associations.
            is_currently_static = getattr(self, 'is_static', False)
            if is_currently_static:
                search_radius = 1
            else:
                search_radius = 2

            for point_lidar in points_to_associate:
                # Transform each point from lidar frame to the world frame based on current pose
                point = T_GI_init @ lidar_imu_extrinsic @ np.append(point_lidar, 1)
                point_world_coords = point[:3]

                association_key = map.get_voxel_key(point_world_coords)

                #find the k nearest points from the global map and fit a plane
                # 4. Fit a plane using SVD
                neighbors = map.query(point_world_coords, radius_voxels=search_radius)
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


                if ratio21 < 0.3:  # optional stricter check, or just an else
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
            correction_applied = False
            self.last_lidar_association_count = len(valid_associations)
            max_iterations = 2
            try:
                P_inv = np.linalg.inv(P_copy)
            except np.linalg.LinAlgError:
                P_inv = np.linalg.pinv(P_copy)

            state_0 = copy_state(state)
            prev_residual_norm = None
            best_state = None
            best_residual_norm = None
            best_H = None
            best_kalman_gain = None
            best_P_new = P_copy

            # --- ITERATED EKF UPDATE ---
            for iter_count in range(max_iterations):
                H_list = []
                residuals = []
                sigmas = []
                T_GI[:3, :3] = state.R
                T_GI[:3, 3] = state.p

                for assoc_type, point_lidar, center, geom_vec in valid_associations:
                    # Transform each point using the continually updated state
                    point = T_GI @ lidar_imu_extrinsic @ np.append(point_lidar, 1)
                    point_coords = point[:3]

                    # Range-dependent measurement noise: sigma = 0.01 + 0.01 * r
                    r_beam = float(np.linalg.norm(point_lidar[:2]))
                    sigma_k = 0.01 + 0.01 * min(max(r_beam, 0.5), 10.0)

                    #find the residual based on the normal and the center point
                    vec = point_coords - center
                    
                    if assoc_type == 'plane':
                        normal = geom_vec
                        res = float(np.dot(normal, vec))
                    else:
                        # Edge/line feature: signed 2D in-plane point-to-line residual
                        direction = geom_vec
                        n_line = np.array([-direction[1], direction[0], 0.0])
                        norm_2d = np.linalg.norm(n_line[:2])
                        if norm_2d > 1e-6:
                            n_line /= norm_2d
                        res = float(np.dot(n_line, vec))
                        normal = n_line


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
                    # Compute a dynamic residual gate based on pose uncertainty
                    innovation_var = float(H_k @ P_copy @ H_k.T + sigma_k**2)
                    k = 4
                    min_gate = 1.20 if search_radius >= 2 else 0.80
                    gate = max(min_gate, k * np.sqrt(max(1e-9, innovation_var)))   # Wide enough to pull back drifted poses without rejection
                    if DEBUG_LIDAR:
                        print("Residual gate value:", gate)
                    if abs(res) > gate:
                        continue
                    residuals.append(res)
                    H_list.append(H_k)
                    sigmas.append(sigma_k)

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

                residual_norm = self.last_lidar_residual_norm
                if prev_residual_norm is not None and residual_norm > prev_residual_norm:
                    if DEBUG_LIDAR:
                        print(
                            f"Residual increased ({residual_norm:.6f} > {prev_residual_norm:.6f}); "
                        )
                    break
                prev_residual_norm = residual_norm
                if DEBUG_LIDAR:
                    print("Residual norm", np.linalg.norm(r))

                sigma_arr = np.asarray(sigmas)
                R_inv = np.diag(1.0 / (sigma_arr**2))
                R_mat = np.diag(sigma_arr**2)
                try:
                    kalman_gain = np.linalg.inv(H.T @ R_inv @ H + P_inv) @ (H.T @ R_inv)
                except np.linalg.LinAlgError:
                    kalman_gain = np.linalg.pinv(H.T @ R_inv @ H + P_inv) @ (H.T @ R_inv)

                # Zero out unobservable dimensions on kalman_gain BEFORE computing dx and BEFORE computing P_new:
                kalman_gain[6:9, :]   = 0.0  # Velocity is unobservable from a position-only LiDAR update
                kalman_gain[9:12, :]  = 0.0  # Prevent gyro bias corruption from scan matching noise
                kalman_gain[12:15, :] = 0.0  # Accel bias is unobservable from LiDAR; lock to calibrated value (leave to ZUPT)
                kalman_gain[15:18, :] = 0.0  # Gravity is frozen/unobservable

                # dx = kalman_gain @ r #error-state vector
                dx_from_prior = state_error(state, state_0)
                dx = kalman_gain @ r - (np.eye(18) - kalman_gain @ H) @ dx_from_prior

                if DEBUG_LIDAR:
                    print(
                        "LiDAR correction candidate:",
                        f"iteration={iter_count + 1}",
                        f"total={np.linalg.norm(dx):.6f}",
                        f"rot={np.linalg.norm(dx[0:3]):.6f}",
                        f"pos={np.linalg.norm(dx[3:6]):.6f}",
                        f"vel={np.linalg.norm(dx[6:9]):.6f}",
                        f"bg={np.linalg.norm(dx[9:12]):.6f}",
                        f"ba={np.linalg.norm(dx[12:15]):.6f}",
                        f"g={np.linalg.norm(dx[15:18]):.6f}",
                        f"median_residual={np.median(np.abs(r)):.6f}",
                    )
                    print("dx:", dx)
                    print("rot:", dx[:3])
                    print("pos:", dx[3:6])
                    print("vel:", dx[6:9])
                    print("bg:", dx[9:12])
                    print("ba:", dx[12:15])
                    print("g:", dx[15:18])

                # Check if the optimization step has converged before state projection
                if np.linalg.norm(dx) < eps:
                    correction_applied = True
                    break

                # Planar 2D robot constraints on state and covariance correction
                kalman_gain[0:2, :] = 0.0  # 2D planar LiDAR cannot observe roll/pitch
                kalman_gain[5, :]   = 0.0  # 2D planar LiDAR cannot observe Z translation
                dx[0:2]   = 0.0  # 2D planar LiDAR cannot observe roll/pitch
                dx[5]     = 0.0  # 2D planar LiDAR cannot observe Z translation
                dx[6:9]   = 0.0  # Velocity is unobservable from position-only LiDAR update
                dx[9:12]  = 0.0
                dx[12:15] = 0.0
                dx[15:18] = 0.0

                if not np.all(np.isfinite(dx)):
                    correction_applied = False
                    break

                # Verified clip sized from pose covariance
                sigma_p = np.sqrt(max(0.0, float(P_copy[3, 3] + P_copy[4, 4])))
                if getattr(self, 'is_static', False):
                    max_position_correction = float(np.clip(3.0 * sigma_p, 0.15, 0.40))
                else:
                    max_position_correction = float(np.clip(3.0 * sigma_p, 0.25, 0.50))
                max_rotation_correction = np.deg2rad(25.0)

                # Clip step magnitude rather than hard-rejecting
                pos_corr = float(np.linalg.norm(dx[3:6]))
                if pos_corr > max_position_correction:
                    dx[3:6] = (dx[3:6] / pos_corr) * max_position_correction

                rot_corr = float(np.linalg.norm(dx[0:3]))
                if rot_corr > max_rotation_correction:
                    dx[0:3] = (dx[0:3] / rot_corr) * max_rotation_correction

                cum_pos = float(np.linalg.norm((state.p + dx[3:6])[:2] - state_0.p[:2]))
                if cum_pos > max_position_correction:
                    dir_cum = (state.p + dx[3:6])[:2] - state_0.p[:2]
                    target_cum = (dir_cum / cum_pos) * max_position_correction
                    dx[3:5] = target_cum - (state.p[:2] - state_0.p[:2])

                theta_rot = dx[0:3]
                state.R = state.R @ exp(theta_rot)
                state.R = reorthonormalize(state.R)   # normalize the R matrix to prevent 
                state.p  += dx[3:6]
                state.p[2] = 0.0
                state.v  += dx[6:9]
                state.v[2] = 0.0
                state.bg += dx[9:12]
                state.ba += dx[12:15]
                state.g[:] = 0.0
                correction_applied = True
                if best_residual_norm is None or residual_norm < best_residual_norm:
                    best_residual_norm = residual_norm
                    best_state = copy_state(state)
                    best_H = H.copy()
                    best_kalman_gain = kalman_gain.copy()
                    # Joseph form covariance update to prevent collapse of P
                    I_KH = np.eye(P_copy.shape[0]) - kalman_gain @ H
                    best_P_new = I_KH @ P_copy @ I_KH.T + kalman_gain @ R_mat @ kalman_gain.T
                    best_P_new = 0.5 * (best_P_new + best_P_new.T)
                if DEBUG_LIDAR:
                    print(
                        "LiDAR correction committed:",
                        f"position={state.p}",
                        f"bg={state.bg}",
                        f"ba={state.ba}",
                        f"g={state.g}",
                        f"R_error={np.linalg.norm(state.R.T @ state.R - np.eye(3)):.3e}",
                        f"det_R={np.linalg.det(state.R):.12f}",
                    )
            if correction_applied and best_state is not None:
                state.R = best_state.R
                state.p = best_state.p
                state.v = best_state.v
                state.bg = best_state.bg
                state.ba = best_state.ba
                state.g = best_state.g
            #Prevent crash when there is no LiDAR update
            if correction_applied and best_kalman_gain is not None and best_H is not None:
                P_new = best_P_new
                self.last_lidar_update_applied = True

            # Generate final world points for the map using the converged state
            T_GI[:3, :3] = state.R
            T_GI[:3, 3] = state.p
            lidar_homo = np.hstack([lidar_points_compensated, np.ones((len(lidar_points_compensated), 1))])
            points_world_final = (T_GI @ lidar_imu_extrinsic @ lidar_homo.T).T[:, :3]

            self.last_lidar_time = time.monotonic()
            return points_world_final, self.last_lidar_update_applied, P_new

        return None, False, P_copy

    def zupt_update(self, sigma_zupt=0.05, adapt_ba=False):
        """
        Zero-velocity pseudo-measurement update. Call when LiDAR scan similarity
        or IMU variance indicates the platform is stationary. Directly modifies self.state/self.P.
        """
        state = self.state
        H = np.zeros((3, 18))
        H[:, 6:9] = np.eye(3)   # measurement model observes velocity directly

        r = -state.v            # residual = measured(0) - predicted velocity

        R_zupt = (sigma_zupt ** 2) * np.eye(3)
        S = H @ self.P @ H.T + R_zupt
        K = self.P @ H.T @ np.linalg.inv(S)

        # Zero out blocked dimensions on K so covariance doesn't artificially shrink:
        K[0:3, :]   = 0.0  # Attitude unobservable from velocity
        K[3:6, :]   = 0.0  # Position unobservable from zero-velocity measurement
        K[9:12, :]  = 0.0  # Gyro bias unobservable from velocity
        if not adapt_ba:
            K[12:15, :] = 0.0  # Accel bias unobservable without long-term tilt reference
        K[14, :]    = 0.0  # Z accel bias is planar locked
        K[15:18, :] = 0.0  # Gravity is frozen

        dx = K @ r
        dx[0:3]   = 0.0
        dx[3:6]   = 0.0
        dx[9:12]  = 0.0
        if not adapt_ba:
            dx[12:15] = 0.0
        dx[14]    = 0.0
        dx[15:18] = 0.0

        if not np.all(np.isfinite(dx)):
            return

        if adapt_ba:
            # Slew-rate limit on accel bias correction per ZUPT cycle (max 1 cm/s^2)
            max_ba_step = 0.01
            dx[12:15] = np.clip(dx[12:15], -max_ba_step, max_ba_step)
            state.ba += dx[12:15]
            # Hard magnitude clamp on ba to prevent runaway
            state.ba = np.clip(state.ba, -1.5, 1.5)

        if DEBUG_LIDAR:
            print(
                "ZUPT correction candidate:",
                f"velocity={np.linalg.norm(r):.6f}",
                f"vel={np.linalg.norm(dx[6:9]):.6f}",
                f"ba={np.linalg.norm(dx[12:15]):.6f}",
            )

        state.p[2] = 0.0  # Planar robot constraint: ground plane height
        state.v   = np.zeros(3)  # Platform is stationary, eliminate residual velocity
        state.g[:] = 0.0

        # Joseph form covariance update
        I_KH = np.eye(18) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_zupt @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        if DEBUG_LIDAR:
            print(
                "ZUPT correction committed:",
                f"position={state.p}",
                f"velocity={state.v}",
                f"bg={state.bg}",
                f"ba={state.ba}",
                f"g={state.g}",
                f"R_error={np.linalg.norm(state.R.T @ state.R - np.eye(3)):.3e}",
                f"det_R={np.linalg.det(state.R):.12f}",
            )