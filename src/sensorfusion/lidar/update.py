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
    scan_count = 0
    update_count = 0
    rate_started = time.monotonic()
    rate_last_report = rate_started
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
            scan_count += 1
            if filter.last_lidar_update_applied:
                update_count += 1
            if points_world is not None:
                filter.state = state
                map.add_points(points_world)

        report_time = time.monotonic()
        if report_time - rate_last_report >= 1.0:
            elapsed = report_time - rate_started
            scan_rate = scan_count / elapsed if elapsed > 0 else 0.0
            update_rate = update_count / elapsed if elapsed > 0 else 0.0
            print(
                f"LiDAR scan rate: {scan_rate:.2f} Hz | EKF update rate: "
                f"{update_rate:.2f} Hz | latest residuals: "
                f"{filter.last_lidar_residual_count}"
            )
            rate_last_report = report_time

        with buffer_lock:
            imu_measurement_buffer.clear()
            imu_measurement_buffer.extend(
                m for m in imu_buffer if m[0] >= lidar_prev_scan_time["time"]
            )
