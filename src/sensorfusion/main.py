from forward import ESIKFStateEstimator
from initialize import Lidar, IMU, CameraSensor
import numpy as np
import time
from lidar.backward import backprop

if __name__ == "__main__":
    filter = ESIKFStateEstimator()
    imu = IMU()
    lidar = Lidar()
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
        imu_measurement_buffer.append((now, imu_data)) #store the timestamp and imu readings for backpropogation of LiDAR points
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
            print("Original LiDAR points: ", scan)
            print("Compensated LiDAR points: ", lidar_points_compensated)
            
        time.sleep(0.01)
        print(state.p)

