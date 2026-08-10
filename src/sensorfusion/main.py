from forward import ESIKFStateEstimator
from initialize import Lidar, IMU, CameraSensor
import numpy as np
import time

if __name__ == "__main__":
    filter = ESIKFStateEstimator()
    imu = IMU()
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

    while(True):
        now = time.time()
        dt = now - prev
        prev = now
        print(dt)

        imu_data = imu.get_readings()
        print("gyro: ", imu_data[0])
        print("accel: ", imu_data[1])
        state, cov = filter.predict(imu_data)
        time.sleep(0.01)
        print(state.p)

    #After forward propogation the LiDAR scan is backpropogated
    