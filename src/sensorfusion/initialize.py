"""
Initialize the sensors and get the initial readings
"""

import time
from rplidar import RPLidar
import adafruit_bno055
import board
import busio
from picamzero import Camera as PiCamera

class IMU:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)

    def get_readings(self):
        gyro = self.sensor.gyro
        accel = self.sensor.acceleration
        return gyro, accel

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


