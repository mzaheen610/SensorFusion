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
            if status >= (3, 3, 3, 3):
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
        # self.wait_for_calibration()
        curr_time = time.time()
        accel_data = []
        gyro_data = []
        
        while(time.time() - curr_time < 10):
            accel = self.sensor.acceleration
            gyro = self.sensor.gyro
            if _is_valid_reading(gyro):
                gyro_data.append(gyro)
            if _is_valid_reading(accel):
                accel_data.append(accel)
            time.sleep(0.01) #sample at 100Hz or else it may read same value multiple times

        if not gyro_data or not accel_data:
            return np.eye(3), np.zeros(3), np.zeros(3)

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
            return np.eye(3), bg, np.zeros(3)
        axis = axis / axis_norm
        delta_theta = axis * angle
        #Finding the initial rotation after SO(3) transform
        R_theta = exp(delta_theta)

        g_world_actual = np.array([0.0, 0.0, -9.81])
        
        # Calculate what gravity should look like in the body frame
        expected_g_body = R_theta.T @ g_world_actual

        aligned_gravity = R_theta @ g_body

        print("Aligned gravity:", aligned_gravity)

        #Estimate the bias
        ba = a_mean- expected_g_body

        return R_theta, bg, ba
    
class Lidar:
    def __init__(self, port='/dev/ttyUSB0'):
        self.port = port
        self._open_lidar()

    def _open_lidar(self):
        self.lidar = RPLidar(self.port, baudrate=256000, timeout=3)
        self.lidar.get_health = lambda: ('Good', 0)
        self.lidar.connect()
        self.lidar.start_motor()
        self.lidar.clean_input()
        time.sleep(2)  # Allow the motor to spin up
        self._scans = self.lidar.iter_scans(max_buf_meas=12000)

    def _reopen_lidar(self):
        try:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()
        except Exception:
            pass

        time.sleep(1)
        self._open_lidar()

    def get_readings(self):
        try:
            scan = next(self._scans)
            return scan
        except Exception as e:
            print(f"Error occurred while fetching LiDAR readings: {e}")
            self._reopen_lidar()
            return None

class CameraSensor:
    def __init__(self):
        self.camera = PiCamera()

    def take_photo(self, path):
        self.camera.take_photo(path)

    def get_frame(self):
        frame = self.camera.capture_array()
        return frame
    


