"""
Implementation of the Lidar thread helper
"""
import copy
import time
import numpy as np
from queue import Empty, Full
from lidar.backward import backprop

# Set to True when inspecting individual scans. Keep False during normal runs
# because console I/O can noticeably reduce throughput on a Raspberry Pi.
DEBUG_LIDAR = False


#Moved the lidar scan acquisition to a seperate thread to limit lidar scan flag mismatch
def lidar_acquisition_thread(lidar, scan_queue):
    """Continuously drain the LiDAR serial stream into a latest-scan queue."""
    while True:
        scan = lidar.get_readings()
        if scan is None:
            # time.sleep(0.01)
            continue

        # Processing must never make the serial reader wait.  Retain only the
        # most recent complete scan when the fusion update falls behind.
        try:
            scan_queue.put_nowait(scan)
        except Full:
            try:
                scan_queue.get_nowait()
            except Empty:
                pass
            scan_queue.put_nowait(scan)


def lidar_thread(state_lock, buffer_lock, filter, map, imu_measurement_buffer,
                 lidar_prev_scan_time, scan_queue):
    """
    Backward propogation
    """
    scan_count = 0
    update_count = 0
    rate_started = time.monotonic()
    rate_last_report = rate_started
    while True:
        scan = scan_queue.get()

        with state_lock:
            state = copy.deepcopy(filter.state)
        # Keep the exact pre-update snapshot so only the correction delta can
        # be merged into the state that IMU prediction advances concurrently.
        state_old = copy.deepcopy(state)
        with buffer_lock:
            imu_buffer = list(imu_measurement_buffer)

        now = time.time()
        lidar_points_compensated = backprop(
            now, lidar_prev_scan_time["time"], state, scan, imu_buffer
        )
        lidar_prev_scan_time["time"] = now
        if DEBUG_LIDAR:
            for i in range(min(5, len(scan))):
                print("Original LiDAR points: ", scan[i])
            for i in range(min(5, len(lidar_points_compensated))):
                print("Compensated LiDAR points: ", lidar_points_compensated[i])
            print("Current position: ", state.p)

        points_world, update_applied = filter.lidar_update(
            scan, state, lidar_points_compensated, map
        )
        with state_lock:
            scan_count += 1
            if update_applied:
                update_count += 1
                # Merge the correction computed from state_old into the latest
                # IMU-predicted state; never replace it with a stale snapshot.
                delta_p = state.p - state_old.p
                delta_v = state.v - state_old.v
                delta_R = state_old.R.T @ state.R
                filter.state.p += delta_p
                filter.state.v += delta_v
                filter.state.R = filter.state.R @ delta_R
                filter.state.bg = state.bg
                filter.state.ba = state.ba
                filter.state.g = state.g
            if points_world is not None:
                map.add_points(points_world)
        #DEBUG THE LIDAR SCAN AND EKF UPDATE RATE  
        report_time = time.monotonic()
        if report_time - rate_last_report >= 1.0:
            elapsed = report_time - rate_started
            scan_rate = scan_count / elapsed if elapsed > 0 else 0.0
            update_rate = update_count / elapsed if elapsed > 0 else 0.0
            if DEBUG_LIDAR:
                print(
                    f"LiDAR scan rate: {scan_rate:.2f} Hz | EKF update rate: "
                    f"{update_rate:.2f} Hz | latest residuals: "
                    f"{filter.last_lidar_residual_count}"
                )
            rate_last_report = report_time
        #clear the imu buffer after the current batch is processed
        with buffer_lock:
            imu_measurement_buffer.clear()
            imu_measurement_buffer.extend(
                m for m in imu_buffer if m[0] >= lidar_prev_scan_time["time"]
            )
