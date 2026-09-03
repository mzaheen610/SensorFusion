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
DEBUG_LIDAR = True


#Moved the lidar scan acquisition to a seperate thread to limit lidar scan flag mismatch
def lidar_acquisition_thread(lidar, scan_queue):
    """Continuously drain the LiDAR serial stream into a latest-scan queue."""
    scan_count = 0
    last_report = time.monotonic()
    while True:
        scan = lidar.get_readings()
        if scan is None:
            # time.sleep(0.01)
            continue

        scan_count += 1
        now = time.monotonic()
        if now - last_report >= 1.0:
            print(
                f"LiDAR acquisition rate: {scan_count / (now - last_report):.1f} "
                f"Hz | points in latest scan: {len(scan)}",
                flush=True,
            )
            scan_count = 0
            last_report = now

        # Processing must never make the serial reader wait.  Retain only the
        # most recent complete scan when the fusion update falls behind.
        try:
            scan_queue.put_nowait(scan)
        except Full:
            try:
                # multiprocessing.Queue uses a feeder thread, so an item may
                # already reserve the one slot before it is readable here.
                scan_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                scan_queue.put_nowait(scan)
            except Full:
                pass


def lidar_acquisition_process(port, scan_queue):
    """Own the serial device in a separate process from fusion work."""
    print("LiDAR acquisition process starting.", flush=True)
    try:
        from initialize import Lidar

        lidar = Lidar(port)
        print("LiDAR acquisition process connected.", flush=True)
        lidar_acquisition_thread(lidar, scan_queue)
    except Exception as error:
        print(
            f"LiDAR acquisition process stopped: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )


def lidar_thread(state_lock, buffer_lock, filter, map, imu_measurement_buffer,
                 lidar_prev_scan_time, scan_queue):
    """
    Backward propogation
    """
    scan_count = 0
    update_count = 0
    empty_compensation_count = 0
    rate_started = time.monotonic()
    rate_last_report = rate_started
    while True:
        try:
            scan = scan_queue.get()

            with state_lock:
                state = copy.deepcopy(filter.state)
                P_copy = copy.deepcopy(filter.P)

            # Keep the exact pre-update snapshot so only the correction delta can
            # be merged into the state that IMU prediction advances concurrently.
            state_old = copy.deepcopy(state)
            with buffer_lock:
                imu_buffer = list(imu_measurement_buffer)

            now = time.time()
            scan_start = time.monotonic()
            lidar_points_compensated = backprop(
                now, lidar_prev_scan_time["time"], state, scan, imu_buffer
            )
            lidar_prev_scan_time["time"] = now

            if lidar_points_compensated.shape[0] == 0:
                empty_compensation_count += 1
                if DEBUG_LIDAR:
                    print(
                        f"LiDAR compensation skipped: scan_points={len(scan)} | "
                        f"imu_buffer={len(imu_buffer)} | "
                        f"scan_age={now - imu_buffer[0][0]:.3f}s"
                        if imu_buffer else
                        f"LiDAR compensation skipped: scan_points={len(scan)} | "
                        "imu_buffer=0",
                        flush=True,
                    )
                continue

            if DEBUG_LIDAR:
                for i in range(min(5, len(scan))):
                    print("Original LiDAR points: ", scan[i])
                for i in range(min(5, len(lidar_points_compensated))):
                    print("Compensated LiDAR points: ", lidar_points_compensated[i])
                print("Current position: ", state.p)

            points_world, update_applied, P_new = filter.lidar_update(
                scan, state, P_copy, lidar_points_compensated, map
            )
            scan_duration = time.monotonic() - scan_start
            if DEBUG_LIDAR:
                print(
                    f"LiDAR update diagnostics: compensated={len(lidar_points_compensated)} | "
                    f"map_points={map.num_points()} | "
                    f"associations={filter.last_lidar_association_count} | "
                    f"applied={update_applied} | "
                    f"residuals={filter.last_lidar_residual_count} | "
                    f"duration={scan_duration:.3f}s",
                    flush=True,
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
                    filter.P = P_new
                if points_world is not None:
                    map.add_points(points_world)
            #DEBUG THE LIDAR SCAN AND EKF UPDATE RATE  
            report_time = time.monotonic()
            if report_time - rate_last_report >= 1.0:
                elapsed = report_time - rate_started
                scan_rate = scan_count / elapsed if elapsed > 0 else 0.0
                update_rate = update_count / elapsed if elapsed > 0 else 0.0
                if DEBUG_LIDAR:
                    interval = report_time - rate_last_report
                    print(
                        f"LiDAR scan rate: {scan_count / interval:.2f} Hz | "
                        f"EKF update rate: {update_count / interval:.2f} Hz | "
                        f"latest residuals: "
                        f"{filter.last_lidar_residual_count} | "
                        f"empty compensation: {empty_compensation_count} | "
                        f"interval: {interval:.2f}s"
                    )
                scan_count = 0
                update_count = 0
                empty_compensation_count = 0
                rate_last_report = report_time
            #clear the imu buffer after the current batch is processed
            with buffer_lock:
                # Cleanly discard old items without touching newly appended ones
                while imu_measurement_buffer and imu_measurement_buffer[0][0] < lidar_prev_scan_time["time"]:
                    imu_measurement_buffer.popleft()

        except Exception as e:
            print(f"[lidar_thread] FATAL: {type(e).__name__}: {e}", flush=True)
            continue