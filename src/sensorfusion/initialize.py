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

class IMU:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)

    def get_readings(self):
        gyro = self.sensor.gyro
        accel = self.sensor.acceleration
        return gyro, accel
    
    def initialize_rotation_gyro(self):
        #Find the initial rotation matrix from the IMU readings
        #Collect 5s of IMU data to get the mean acceleration
        curr_time = time.time()
        accel_data = []
        gyro_data = []
        
        while(time.time() - curr_time < 5):
            accel = self.sensor.acceleration
            gyro = self.sensor.gyro
            if gyro is not None:
                gyro_data.append(gyro)
            if accel is not None:
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


