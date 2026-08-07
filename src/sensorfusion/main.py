from forward import ESIKFStateEstimator
from initialize import Lidar, IMU, CameraSensor
import numpy as np
import time

if __name__ == "__main__":
    filter = ESIKFStateEstimator()
    imu = IMU()
    filter.state.R, filter.state.bg = imu.initialize_rotation_gyro()
    initial_covariance = 100 * np.eye(18)

    while(True):
        imu_data = imu.get_readings()
        print("gyro: ", imu_data[0])
        print("accel: ", imu_data[1])
        state, cov = filter.predict(imu_data)
        time.sleep(1)
        print(state.p)