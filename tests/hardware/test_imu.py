#!/usr/bin/env python3

import time
import board
import busio
import adafruit_bno055

def main():
    print("Initializing BNO055...")

    # Initialize I2C
    i2c = busio.I2C(board.SCL, board.SDA)

    # Initialize sensor
    sensor = adafruit_bno055.BNO055_I2C(i2c)

    # Give the sensor some time to start
    time.sleep(1)

    print("BNO055 initialized successfully.\n")

    while True:
        print("=" * 60)

        # Calibration status
        try:
            sys_cal, gyro_cal, accel_cal, mag_cal = sensor.calibration_status
            print(f"Calibration:")
            print(f"  System : {sys_cal}/3")
            print(f"  Gyro   : {gyro_cal}/3")
            print(f"  Accel  : {accel_cal}/3")
            print(f"  Mag    : {mag_cal}/3")
        except Exception as e:
            print("Calibration:", e)

        # Gyroscope (rad/s)
        try:
            print(f"Gyroscope (rad/s):          {sensor.gyro}")
        except Exception as e:
            print("Gyroscope:", e)

        # Raw acceleration (includes gravity)
        try:
            print(f"Acceleration (m/s²):        {sensor.acceleration}")
        except Exception as e:
            print("Acceleration:", e)

        # Linear acceleration (gravity removed)
        try:
            print(f"Linear Accel (m/s²):        {sensor.linear_acceleration}")
        except Exception as e:
            print("Linear Acceleration:", e)

        # Gravity vector
        try:
            print(f"Gravity (m/s²):             {sensor.gravity}")
        except Exception as e:
            print("Gravity:", e)

        # Magnetometer
        try:
            print(f"Magnetometer (uT):          {sensor.magnetic}")
        except Exception as e:
            print("Magnetometer:", e)

        # Euler angles
        try:
            print(f"Euler Angles (deg):         {sensor.euler}")
        except Exception as e:
            print("Euler:", e)

        # Quaternion
        try:
            print(f"Quaternion:                {sensor.quaternion}")
        except Exception as e:
            print("Quaternion:", e)

        # Temperature
        try:
            print(f"Temperature (°C):           {sensor.temperature}")
        except Exception as e:
            print("Temperature:", e)

        print()
        time.sleep(0.5)


if __name__ == "__main__":
    main()