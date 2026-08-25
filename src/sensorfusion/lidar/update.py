"""
Implementation of the Lidar thread helper
"""
import copy
import time
import numpy as np
from lidar.backward import backprop

def lidar_thread(state_lock, buffer_lock, filter, lidar, map, imu_measurement_buffer, lidar_prev_scan_time):
    """
    Backward propogation
    """
    while True:
        scan = lidar.get_readings()
        if scan is None:
            time.sleep(0.01)
            continue

        with state_lock:
            state = copy.deepcopy(filter.state)
        with buffer_lock:
            imu_buffer = list(imu_measurement_buffer)

        now = time.time()
        lidar_points_compensated = backprop(
            now, lidar_prev_scan_time["time"], state, scan, imu_buffer
        )
        lidar_prev_scan_time["time"] = now
        for i in range(min(5, len(scan))):
            print("Original LiDAR points: ", scan[i])
        for i in range(min(5, len(lidar_points_compensated))):
            print("Compensated LiDAR points: ", lidar_points_compensated[i])
        print("Current position: ", state.p)

        with state_lock:
            points_world = filter.lidar_update(
                scan, state, lidar_points_compensated, map
            )
            if points_world is not None:
                filter.state = state
                map.add_points(points_world)

        with buffer_lock:
            imu_measurement_buffer.clear()
            imu_measurement_buffer.extend(
                m for m in imu_buffer if m[0] >= lidar_prev_scan_time["time"]
            )