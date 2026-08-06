from forward import ESIKFStateEstimator
from initialize import Lidar, IMU, CameraSensor
import numpy as np

if __name__ == "__main__":
    filter = ESIKFStateEstimator()
    initial_state = np.zeros(18)
    initial_covariance = 100 * np.eye(18)
    imu = IMU()

    while(True):
        imu_data = imu.get_readings()
        state, cov = filter.predict(imu_data)
        print(state, cov)