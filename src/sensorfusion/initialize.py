"""
Initialize the sensors and get the initial readings
"""

import time
from rplidar import RPLidar
from adafruit_bno055 import BNO055
from picamzero import Camera

class IMU:
    def __init__(self):
        self.sensor = BNO055.BNO055()

    def get_readings(self):
        return self.sensor.read_euler()

class Lidar:
    def __init__(self, port='/dev/ttyUSB0'):
        self.lidar = RPLidar(port)

    def get_readings(self):
        return list(self.lidar.iter_scans())

class Camera:
    def __init__(self):
        self.camera = Camera()

    def take_photo(self, path):
        self.camera.take_photo(path)


