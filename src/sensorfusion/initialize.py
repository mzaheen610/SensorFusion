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
        # accel = self.sensor.acceleration
        linear_accel = self.sensor.linear_acceleration
        if gyro is not None and linear_accel is not None:
            return np.array(gyro), np.array(linear_accel)
        time.sleep(0.001)
    
    def initialize_rotation_gyro(self):
        """
        Finding the initial rotation based on observed 
        gravity and theoretical value.
        Also estimate accelerometer bias from observations.
        """
        #Find the initial rotation matrix from the IMU readings
        #Collect 5s of IMU data to get the mean acceleration
        self.wait_for_calibration()
        time.sleep(5) #after calibration wait for the sensor to be placed static
        curr_time = time.time()
        # accel_data = []
        gyro_data = []
        gravity_data = []
        linear_accel_data = []
        
        while(time.time() - curr_time < 10):
            # accel = self.sensor.acceleration
            gyro = self.sensor.gyro
            gravity = self.sensor.gravity
            linear_accel = self.sensor.linear_acceleration #accel with gravity removed onboard the sensor
            if _is_valid_reading(gyro):
                gyro_data.append(gyro)
            if _is_valid_reading(linear_accel):
                linear_accel_data.append(linear_accel)
            if _is_valid_reading(gravity):
                gravity_data.append(gravity)
            time.sleep(0.01) #sample at 100Hz or else it may read same value multiple times

        if not gyro_data or not gravity_data:
            return np.eye(3), np.zeros(3), np.zeros(3)

        #Estimate gyro bias
        bg = np.mean(gyro_data, axis=0)

        # a_mean = np.mean(accel_data, axis=0)
        g_mean = np.mean(gravity_data, axis=0)
        g_body = g_mean / np.linalg.norm(g_mean) #direction of gravity vector wrt IMU body
        g_world = np.array([0,0,-1])
        #Find the axis of rotation by taking cross product of the two vectors
        axis = np.cross(g_body, g_world)
        #Find angle of rotation by taking dot product of the two vecs
        angle = np.arccos(np.clip(np.dot(g_body, g_world), -1.0, 1.0))

        #Normalize the axis to get only the direction
        axis_norm = np.linalg.norm(axis)

        #Estimate bias from the stationary linear accel data
        ba = np.mean(linear_accel_data, axis=0) if linear_accel_data else np.zeros(3)
        print("Residual linear_acceleration bias (should be small, e.g. <2):", ba)

        if axis_norm < 1e-8:
            return np.eye(3), bg, np.zeros(3)
        axis = axis / axis_norm
        delta_theta = axis * angle
        #Finding the initial rotation after SO(3) transform
        R_theta = exp(delta_theta)

        # g_world_actual = np.array([0.0, 0.0, -9.81])
        
        # # Calculate what gravity should look like in the body frame
        # expected_g_body = R_theta.T @ g_world_actual

        # aligned_gravity = R_theta @ g_body

        # print("Aligned gravity:", aligned_gravity)

        # #Estimate the bias
        # ba = a_mean- expected_g_body

        return R_theta, bg, ba
    
class Lidar:
    def __init__(self, port='/dev/ttyUSB0', max_reconnect_attempts=5):
        self.port = port
        self.max_reconnect_attempts = max_reconnect_attempts
        self._open_lidar(spinup_delay=4.0)

    def _open_lidar(self, spinup_delay=4.0):
        self.lidar = RPLidar(self.port, baudrate=256000, timeout=3)
        self.lidar.get_health = lambda: ('Good', 0)
        self.lidar.connect()
        self.lidar.stop_motor()
        self.lidar.start_motor()
        time.sleep(spinup_delay)
        self.lidar.clean_input()
        self._scans = self.lidar.iter_scans(max_buf_meas=12000)

    def _close_lidar(self):
        # Each teardown step gets its own try/except so one failure
        # doesn't skip the others (especially disconnect(), which
        # frees the serial port for the next open).
        for step in (self.lidar.stop, self.lidar.stop_motor,
                     self.lidar.reset, self.lidar.disconnect):
            try:
                step()
            except Exception as e:
                print(f"Error during LiDAR teardown ({step.__name__}): "
                      f"{type(e).__name__}: {e}")

    def _reopen_lidar(self):
        self._close_lidar()

        # The motor is usually still warm during a runtime reconnect, so a
        # short delay is normally enough and avoids reducing the scan
        # stream to ~0.2 Hz. If the motor actually spun down (e.g. a
        # power/USB drop), this may not be long enough — that's a
        # known tradeoff, not an oversight.
        try:
            self._open_lidar(spinup_delay=0.5)
        except Exception as e:
            print(f"Error while reopening LiDAR: {type(e).__name__}: {e}")
            raise
    def get_readings(self):
        attempts = 0
        while attempts < self.max_reconnect_attempts:
            try:
                return next(self._scans)
            except Exception as e:
                error_msg = str(e)
                print(f"LiDAR exception: {type(e).__name__}: {error_msg}")
                
                # Check for parsing desyncs (dropped bytes)
                if any(x in error_msg for x in ["Check bit", "descriptor", "mismatch"]):
                    # 1. Pause the data stream (motor stays running)
                    try:
                        self.lidar.stop()
                    except Exception:
                        pass 
                    # 2. Allow OS serial buffer to catch up, then flush
                    time.sleep(0.05) 
                    self.lidar.clean_input()
                    # 3. Restart the scan generator
                    self._scans = self.lidar.iter_scans(max_buf_meas=12000)
                    time.sleep(0.05) # Brief pause before next read
                    continue
                # Hard reset for actual hardware disconnects
                attempts += 1
                try:
                    self._reopen_lidar()
                except Exception:
                    time.sleep(min(2 ** attempts, 30))
                    
        print(f"Giving up after {self.max_reconnect_attempts} reconnect attempts.")
        return None

class CameraSensor:
    def __init__(self):
        self.camera = PiCamera()

    def take_photo(self, path):
        self.camera.take_photo(path)

    def get_frame(self):
        frame = self.camera.capture_array()
        return frame
    

