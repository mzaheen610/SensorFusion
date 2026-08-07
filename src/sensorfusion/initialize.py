"""
Initialize the sensors and get the initial readings
"""

import time
from rplidar import RPLidar
import adafruit_bno055
import board
import busio
from picamzero import Camera as PiCamera
import numpy as np
from utils.so3_rotation import exp


def _is_valid_reading(reading):
    if reading is None:
        return False
    try:
        values = np.asarray(reading, dtype=float)
    except (TypeError, ValueError):
        return False
    return np.all(np.isfinite(values))


def _get_calibration_status(sensor):
    status = getattr(sensor, "calibration_status", None)
    if callable(status):
        status = status()

    if status is None:
        calibrated = getattr(sensor, "calibrated", None)
        if calibrated is True:
            return (3, 3, 3, 3)
        return None

    try:
        values = tuple(int(value) for value in status)
    except TypeError:
        return None

    if len(values) == 4:
        return values

    return None

class IMU:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)

    def wait_for_calibration(self):
        last_status = None
        print("Waiting for BNO055 calibration...")

        while True:
            status = _get_calibration_status(self.sensor)
            if status == (3, 3, 3, 3):
                print("BNO055 calibration complete.")
                return status

            if status != last_status:
                print("Calibration status:", status)
                last_status = status

            time.sleep(0.1)

    def get_readings(self):
        gyro = self.sensor.gyro
        accel = self.sensor.acceleration
        if gyro is not None and accel is not None:
            return np.array(gyro), np.array(accel)
        time.sleep(0.001)
    
    def initialize_rotation_gyro(self):
        #Find the initial rotation matrix from the IMU readings
        #Collect 5s of IMU data to get the mean acceleration
        self.wait_for_calibration()
        curr_time = time.time()
        accel_data = []
        gyro_data = []
        
        while(time.time() - curr_time < 5):
            accel = self.sensor.acceleration
            gyro = self.sensor.gyro
            if _is_valid_reading(gyro):
                gyro_data.append(gyro)
            if _is_valid_reading(accel):
                accel_data.append(accel)
            time.sleep(0.01) #sample at 100Hz or else it may read same value multiple times

        if not gyro_data or not accel_data:
            return np.eye(3), np.zeros(3)

        #Estimate gyro bias
        bg = np.mean(gyro_data, axis=0)

        a_mean = np.mean(accel_data, axis=0)
        g_body = a_mean / np.linalg.norm(a_mean) #direction of gravity vector wrt IMU body
        g_world = np.array([0,0,-1])
        #Find the axis of rotation by taking cross product of the two vectors
        axis = np.cross(g_body, g_world)
        #Find angle of rotation by taking dot product of the two vecs
        angle = np.arccos(np.clip(np.dot(g_body, g_world), -1.0, 1.0))

        #Normalize the axis to get only the direction
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return np.eye(3), bg
        axis = axis / axis_norm
        delta_theta = axis * angle
        #Finding the initial rotation after SO(3) transform
        R_theta = exp(delta_theta)
        return R_theta, bg
    
class Lidar:
    def __init__(self, port='/dev/ttyUSB0'):
        self.lidar = RPLidar(port)

    def get_readings(self):
        return list(self.lidar.iter_scans())

class CameraSensor:
    def __init__(self):
        self.camera = PiCamera()

    def take_photo(self, path):
        self.camera.take_photo(path)


