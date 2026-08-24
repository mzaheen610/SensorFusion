from forward import ESIKFStateEstimator
from initialize import Lidar, IMU, CameraSensor
import numpy as np
import time
from lidar.backward import backprop
from map import Map
from utils.so3_rotation import skew, exp
from utils.projections import project_points_to_frame, project_points_world
if __name__ == "__main__":
    filter = ESIKFStateEstimator()
    imu = IMU()
    lidar = Lidar()
    cam = CameraSensor()
    map = Map()
    time.sleep(1.0)   # Let BNO055 finish initializing
    filter.state.R, filter.state.bg, filter.state.ba = imu.initialize_rotation_gyro()
    initial_covariance = 100 * np.eye(18)

    print("Initial R:")
    print(filter.state.R)

    u = imu.get_readings()
    a_body = np.array(u[1])
    a_world = filter.state.R @ a_body

    print("Body accel :", a_body)
    print("World accel:", a_world)
    print("Bias corrected acceleration:", (a_world - filter.state.ba))

    prev = time.time()
    imu_state_buffer = []
    imu_measurement_buffer = []
    lidar_prev_scan_time = time.time()

    while(True):
        now = time.time()
        dt = now - prev
        prev = now
        print("IMU prdiction frequency:", (1/dt))

        """
        Forward propogation
        """
        imu_data = imu.get_readings()
        imu_measurement_buffer.append((now, imu_data[0], imu_data[1])) #store the timestamp and imu readings for backpropogation of LiDAR points
        print("gyro: ", imu_data[0])
        print("accel: ", imu_data[1])
        #After getting the IMU readings, we perform the forward propagation of the state using the ESIKF filter
        state, cov = filter.predict(imu_data, dt)

        imu_state = (now, state) #store the timestamp and state for backpropogation of LiDAR points
        imu_state_buffer.append(imu_state)

        """
        Backward propogation
        """
        #After forward propogation the LiDAR points are backpropogated when the scan arrives
        scan = lidar.get_readings()
        if scan is not None:
            now = time.time()
            lidar_points_compensated = backprop(now, lidar_prev_scan_time, state, scan, imu_measurement_buffer)
            lidar_prev_scan_time = now
            for i in range(5):
                print("Original LiDAR points: ", scan[i])
            for i in range(5):
                print("Compensated LiDAR points: ", lidar_points_compensated[i])
            print("Current position: ", state.p)

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
                print("Not enough points in the map")
                continue

            if map.is_empty():
                points_world = []
                T_GI[:3, :3] = state.R
                T_GI[:3, 3] = state.p
                for point_lidar in lidar_points_compensated:
                    point = T_GI @ lidar_imu_extrinsic @ np.append(point_lidar, 1)
                    points_world.append(point[:3])
                map.add_points(points_world)
                
                # Clean up IMU buffer and skip EKF update
                imu_measurement_buffer = [m for m in imu_measurement_buffer if m[0] >= lidar_prev_scan_time]
                continue # Skips to the camera logic

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
                    center = np.mean(neighbors, axis=0) 
                    centered_neighbors = neighbors - center
                    #find the normal to the plane based on the SVD
                    _, _, vh = np.linalg.svd(centered_neighbors)
                    normal = vh[-1, :]  # Plane normal vector
                    #find the residual based on the normal and the center point
                    vec = point - center
                    res = np.dot(normal, vec)
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
                    print("No valid LiDAR points for EKF update")
                    break

                H = np.vstack(H_list)
                r = np.array(residuals)

                error_pred = cov @ H.T
                sigma_lidar = 0.02
                # lidar_sensor_noise = sigma_lidar**2 * np.eye(len(r)) #error in the lidar measurement
                # error_meas = H @ cov @ H.T + lidar_sensor_noise
                # kalman_gain = error_pred @ np.linalg.inv(error_meas) #calculating the Kalman Gain

                R_inv = (1.0 / sigma_lidar**2) * np.eye(len(r)) 
                P_inv = np.linalg.inv(cov)
                kalman_gain = np.linalg.inv(H.T @ R_inv @ H + P_inv) @ (H.T @ R_inv)

                dx = kalman_gain @ r #error-state vector

                if np.linalg.norm(dx) < eps:
                    break

                theta_rot = dx[0:3]
                state.R = state.R @ exp(theta_rot)
                state.p  += dx[3:6]
                state.v  += dx[6:9]
                state.bg += dx[9:12]
                state.ba += dx[12:15]
                state.g  += dx[15:18]
            I = np.eye(cov.shape[0])
            cov = (I - kalman_gain @ H) @ cov #covariance update

            #add the lidar points to the map after the lidar based update is done
            map.add_points(points_world)
            #buffer maintenance
            imu_measurement_buffer = [m for m in imu_measurement_buffer if m[0] >= lidar_prev_scan_time]
        #Get the camera scan at 10Hz
        frame = cam.get_frame()
        
        # #find visual map points for the current frame based on current pose and current lidar scan
        # visual_map_points = map.query_visible_voxels(scan) #visible voxel query

        # #project lidar points to the current camera frame (u,v)
        # R_CI = np.eye(3) 
        # t_CI = np.eye(3)

        # T_CI = np.eye(4) #dummy camera imu extrinsics, real values have to be calibrated later
        # T_CI[:3, :3] = R_CI
        # T_CI[:3, 3] = t_CI

        # projected_points = project_points_to_frame(visual_map_points, T_GI, T_CI)
        # #get the 8x8 pixel patch surrounding the current lidar point
        # #attach the patch to the lidar map point

        # print(frame)
        print("Current state (x,y,z): ", state.p)


