#!/usr/bin/env python3
"""Test if LiDAR is receiving valid scan data"""

import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')

import time
import numpy as np
from rplidar import RPLidar

LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = 256000

print("[TEST] Attempting LiDAR connection...")
try:
    lidar = RPLidar(LIDAR_PORT, baudrate=LIDAR_BAUD, timeout=3)
    lidar.get_health = lambda: ('Good', 0)  # Monkey-patch
    lidar.connect()
    print("[TEST] Connected to LiDAR")
    
    lidar.start_motor()
    time.sleep(2)
    lidar.clean_input()
    
    print("[TEST] Reading scans...")
    scan_count = 0

    for scan in lidar.iter_scans(max_buf_meas=5000):
        scan_count += 1
        for m in scan:
            x = m[2]* np.cos(m[1]) #rcos(theta)
            y = m[2]* np.sin(m[1]) #rsin(theta)
            print("Points (x,y): ", x, y)
        angles = np.array([m[1] for m in scan])
        ranges = np.array([m[2] / 1000.0 for m in scan])
        print(f"[SCAN {scan_count}] Points: {len(scan)}, Angle range: {angles.min():.1f}-{angles.max():.1f}°, Range: {ranges.min():.2f}-{ranges.max():.2f}m")

        if scan_count >= 30:
            break
    
    lidar.stop_motor()
    lidar.stop()
    print("[TEST] SUCCESS - LiDAR is working!")
    lidar.disconnect()

except Exception as e:
    print(f"[TEST] FAILED - {e}")
    lidar.stop()
    lidar.stop_motor()
    sys.exit(1)
