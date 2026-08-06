from forward import ESIKFStateEstimator
from initialize import Lidar, IMU, Camera
import numpy as np

if __name__ == "main":
    filter = ESIKFStateEstimator()
    initial_state = np.zeros(18)
    initial_covariance = 100 * np.eye(18)
    while(True):
        imu_data = IMU.get_readings()
        state, cov = filter.predict(imu_data)
        print(state, cov)