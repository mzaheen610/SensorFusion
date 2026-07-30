#!/usr/bin/env python3
"""Quick test for IMU imports"""
import sys
import time
print(f"Python path: {sys.path[:3]}")

try:
    import board, busio
    from adafruit_icm20x import ICM20948
    print("[SUCCESS] IMU libraries imported successfully!")
    IMU_ADDR  = 0x69     
    DT_IMU = 1       
    #test imu data
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        imu = ICM20948(i2c, address=IMU_ADDR)
        while True:
            a_x, a_y, a_z = imu.acceleration
            g_x, g_y, g_z = imu.gyro
            time.sleep(DT_IMU)
            print("Acceleration:", a_x, a_y, a_z)
            print("Gyro:", g_x,g_y,g_z)
    except Exception as err:
        print(f"Failed {err}")
except ImportError as e:
    print(f"[FAILED] {e}")
    sys.exit(1)
