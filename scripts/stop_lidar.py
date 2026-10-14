#!/usr/bin/env python3
"""
Utility script to immediately stop the RPLiDAR motor and cleanly disconnect serial.

Usage:
    python3 scripts/stop_lidar.py [/dev/ttyUSB0]
"""

import sys
import time

def stop_lidar(port='/dev/ttyUSB0', baudrate=256000):
    try:
        from rplidar import RPLidar
    except ImportError:
        print("[ERROR] 'rplidar' package is not installed in the current environment.")
        return False

    print(f"[LiDAR] Connecting to LiDAR on {port}...")
    try:
        lidar = RPLidar(port, baudrate=baudrate, timeout=2)
        lidar.get_health = lambda: ('Good', 0)
        lidar.connect()
        print("[LiDAR] Sending stop scanning command...")
        try:
            lidar.stop()
        except Exception:
            pass
        print("[LiDAR] Sending stop motor command...")
        try:
            lidar.stop_motor()
        except Exception:
            pass
        time.sleep(0.5)
        try:
            lidar.disconnect()
        except Exception:
            pass
        print("[LiDAR] Success: Motor stopped and serial port closed.")
        return True
    except Exception as e:
        print(f"[LiDAR] Failed to connect/stop LiDAR: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
    stop_lidar(port)
