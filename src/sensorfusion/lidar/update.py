"""
Implementation of the Lidar thread helper
"""
import copy
import time
import numpy as np
from backward import backprop

def lidar_thread(state_lock, buffer_lock, filter, lidar):
    """
    Backward propogation
    """
    #After forward propogation the LiDAR points are backpropogated when the scan arrives
    scan = lidar.get_readings()
    with state_lock:
        state = copy.deepcopy(filter.state)
    if scan is not None:
        now = time.time()
        lidar_points_compensated = backprop(now, lidar_prev_scan_time, state, scan, imu_measurement_buffer)
        lidar_prev_scan_time = now
        for i in range(5):
            print("Original LiDAR points: ", scan[i])
        for i in range(5):
            print("Compensated LiDAR points: ", lidar_points_compensated[i])
        print("Current position: ", state.p)

    """
    Lidar Update
    """
    points_world = filter.lidar_update(scan, filter.state, lidar_points_compensated )
    #add the lidar points to the map after the lidar based update is done
    map.add_points(points_world)

    #buffer maintenance
    with buffer_lock:
        imu_measurement_buffer = [m for m in imu_measurement_buffer if m[0] >= lidar_prev_scan_time]