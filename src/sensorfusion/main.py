from forward import ESIKFStateEstimator
from initialize import Lidar, IMU, CameraSensor
import numpy as np
import time
from lidar.backward import backprop
from map import Map
from utils.so3_rotation import skew

if __name__ == "__main__":
    filter = ESIKFStateEstimator()
    imu = IMU()
    lidar = Lidar()
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

    prev = time.time()
    imu_state_buffer = []
    imu_measurement_buffer = []
    lidar_prev_scan_time = 0.0

    while(True):
        now = time.time()
        dt = now - prev
        prev = now
        print(dt)

        imu_data = imu.get_readings()
        imu_measurement_buffer.append((now, imu_data[0], imu_data[1])) #store the timestamp and imu readings for backpropogation of LiDAR points
        print("gyro: ", imu_data[0])
        print("accel: ", imu_data[1])
        #After getting the IMU readings, we perform the forward propagation of the state using the ESIKF filter
        state, cov = filter.predict(imu_data)

        imu_state = (now, state) #store the timestamp and state for backpropogation of LiDAR points
        imu_state_buffer.append(imu_state)

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
            print("Compensated LiDAR points: ", lidar_points_compensated)

        """When the LiDAR scan is motion compensated, do the residual computation 
            and update the state using the ESIKF filter
        """
        #first transform the LiDAR points to the world frame using the current state
        lidar_imu_extrinsic = np.eye(4) #assuming the LiDAR and IMU are co-located
        global_imu_extrinsic = np.eye(4) #assuming the IMU is at the origin of the world frame
        T_GI = np.eye(4) #homogeneous transformation matrix built from current ESIKF state at the scan end time t_k.
        T_GI[:3, :3] = state.R
        T_GI[:3, 3] = state.p
        points_world = []
        for point in lidar_points_compensated:
            # Transform each point from lidar frame to the world frame based on current pose
            point_world = T_GI @ lidar_imu_extrinsic @ np.append(point, 1)
            points_world.append(point_world[:3])  # Extract the 3D coordinates

        #Compute the residuals between each point and the nearest plane in the world map
        H_list = []
        residuals = []
        total_res = 0
        state_updated = np.array()
        eps = 0.01
        while(state - state_updated < eps):
            for point in points_world:
                #find the k nearest points from the global map and fit a plane
                # 4. Fit a plane using SVD
                neighbors = map.query(point, 10)
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
                H_pos = normal 
                H_rot = -normal @ state.R @ skew(point)
                H_k = np.hstack([
                    H_rot,
                    H_pos,
                    np.zeros(3),  # velocity
                    np.zeros(3),  # gyro bias
                    np.zeros(3),  # accel bias
                    np.zeros(3),  # gravity  
                ])      
                H_list.append(H_k)
                
            error_pred = cov @ H_k.T 
            error_meas = H_k @ cov @ H_k.T
            lidar_sensor_noise = np.array() #error in the lidar measurement
            kalman_gain = error_pred / (error_meas)

            state_updated = state + kalman_gain(residuals)

        time.sleep(0.01)
        print(state.p)

