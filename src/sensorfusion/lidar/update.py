"""
Implementation of the Lidar thread helper
"""
import copy
import time
import numpy as np
from queue import Empty, Full
from lidar.backward import backprop
from utils.similarity import scan_similarity, scan_to_bins
from utils.so3_rotation import log, reorthonormalize
# Set to True when inspecting individual scans. Keep False during normal runs
# because console I/O can noticeably reduce throughput on a Raspberry Pi.
DEBUG_LIDAR = False


#Moved the lidar scan acquisition to a seperate thread to limit lidar scan flag mismatch
def lidar_acquisition_thread(lidar, scan_queue, camera_scan_queue=None):
    """Continuously drain the LiDAR serial stream into a latest-scan queue."""
    scan_count = 0
    last_report = time.monotonic()
    prev_scan_timestamp = time.monotonic()
    while True:
        scan = lidar.get_readings()
        if scan is None:
            time.sleep(0.05)
            continue

        scan_timestamp = time.monotonic()
        scan_count += 1
        scan_start_time = prev_scan_timestamp
        prev_scan_timestamp = scan_timestamp
        now = scan_timestamp
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
        item = (scan_timestamp, scan, scan_start_time)
        try:
            scan_queue.put_nowait(item)
        except Full:
            try:
                # multiprocessing.Queue uses a feeder thread, so an item may
                # already reserve the one slot before it is readable here.
                scan_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                scan_queue.put_nowait(item)
            except Full:
                pass

        # if camera_scan_queue is not None:
        #     try:
        #         camera_scan_queue.put_nowait((scan_timestamp, scan))
        #     except Full:
        #         try:
        #             camera_scan_queue.get_nowait()
        #             camera_scan_queue.put_nowait((scan_timestamp, scan))
        #         except (Empty, Full):
        #             pass


def lidar_acquisition_process(port, scan_queue, camera_scan_queue=None):
    """Own the serial device in a separate process from fusion work."""
    import signal
    print("LiDAR acquisition process starting.", flush=True)
    lidar = None

    def _sig_handler(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except (ValueError, AttributeError):
        pass

    try:
        from initialize import Lidar

        lidar = Lidar(port)
        print("LiDAR acquisition process connected.", flush=True)
        lidar_acquisition_thread(lidar, scan_queue, camera_scan_queue)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as error:
        print(
            f"LiDAR acquisition process stopped: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
    finally:
        if lidar is not None:
            print("\n[LiDAR] Stopping motor and disconnecting...", flush=True)
            try:
                lidar.stop_motor()
            except Exception:
                pass
            try:
                lidar.stop()
            except Exception:
                pass
            print("[LiDAR] Motor stopped cleanly.", flush=True)


def lidar_thread(state_lock, buffer_lock, filter, map, imu_measurement_buffer,
                 imu_state_buffer, lidar_prev_scan_time, scan_queue, camera_scan_queue=None):
    """
    Backward propogation
    """
    scan_count = 0
    update_count = 0
    empty_compensation_count = 0
    rate_started = time.monotonic()
    rate_last_report = rate_started

    prev_scan_bins = None
    ZUPT_CONSECUTIVE_REQUIRED = 2
    static_count = ZUPT_CONSECUTIVE_REQUIRED
    static_anchor_pos = None
    static_anchor_R = None
    ZUPT_DIST_THRESHOLD_MM = 45  # mm (85th percentile threshold)
    last_committed_time = None
    last_committed_pos = None
    last_map_pos = None
    last_map_heading = None
    last_turn_time = 0.0
    while True:
        try:
            try:
                scan_item = scan_queue.get(timeout=1.0)
            except Empty:
                # Watchdog: no scan received for 1.0s (sensor disconnect or queue stall)
                with state_lock:
                    filter.is_static = True
                    filter.state.v[:] = 0.0
                continue

            filter.last_lidar_time = time.monotonic()
            if len(scan_item) == 3:
                scan_timestamp, scan, scan_start_time = scan_item
            else:
                scan_timestamp, scan = scan_item
                scan_start_time = scan_timestamp - 0.08

            scan_duration = scan_timestamp - scan_start_time
            if not (0.05 < scan_duration < 0.25):
                scan_start_time = scan_timestamp - 0.08

            if time.monotonic() - scan_timestamp > 0.8:
                if DEBUG_LIDAR:
                    print(
                        f"LiDAR compensation skipped: processing_age="
                        f"{time.monotonic() - scan_timestamp:.3f}s",
                        flush=True,
                    )
                lidar_prev_scan_time["time"] = scan_timestamp
                continue

            with buffer_lock:
                imu_buffer = list(imu_measurement_buffer)
                state_item = next(
                    (
                        item for item in reversed(imu_state_buffer)
                        if item[0] <= scan_timestamp
                    ),
                    None,
                )

            if state_item is None:
                continue
            state_timestamp, state, P_copy = state_item

            # Keep the exact pre-update snapshot so only the correction delta can
            # be merged into the latest state that IMU prediction advances.
            state_old = copy.deepcopy(state)

            scan_start = time.monotonic()
            lidar_points_compensated = backprop(
                scan_timestamp,
                scan_start_time,
                state,
                scan,
                imu_buffer,
            )
            lidar_prev_scan_time["time"] = scan_timestamp

            if lidar_points_compensated.shape[0] == 0:
                empty_compensation_count += 1
                if DEBUG_LIDAR:
                    print(
                        f"LiDAR compensation skipped: scan_points={len(scan)} | "
                        f"imu_buffer={len(imu_buffer)} | "
                        f"scan_age={scan_timestamp - imu_buffer[0][0]:.3f}s"
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

            if camera_scan_queue is not None:
                try:
                    camera_scan_queue.put_nowait((scan_timestamp, scan))
                except Full:
                    try:
                        camera_scan_queue.get_nowait()
                        camera_scan_queue.put_nowait((scan_timestamp, scan))
                    except (Empty, Full):
                        pass

            scan_duration = time.monotonic() - scan_start
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

                # --- ZUPT and motion detection check  ---
                current_bins = scan_to_bins(scan)
                is_imu_static = False
                gyro_max = 0.0
                accel_var = 0.0
                recent_imu = []
                sim = None

                if imu_buffer:
                    recent_imu = [
                        m for m in imu_buffer
                        if scan_timestamp - 0.30 <= m[0] <= scan_timestamp + 0.05
                    ]
                    if len(recent_imu) < 3:
                        recent_imu = imu_buffer[-10:]
                    if len(recent_imu) >= 3:
                        gyros = np.array([m[1] for m in recent_imu])
                        accels = np.array([m[2] for m in recent_imu])
                        gyro_max = float(np.max(np.linalg.norm(gyros, axis=1)))
                        accel_var = float(np.var(np.linalg.norm(accels, axis=1)))
                        # Static only if angular speed <= 0.12 rad/s and accel variance <= 0.40
                        if gyro_max <= 0.12 and accel_var <= 0.40:
                            is_imu_static = True

                if prev_scan_bins is not None:
                    sim = scan_similarity(current_bins, prev_scan_bins)
                    print("Scan similarity between scans is:", sim)

                    # Multi-modal stationary check:
                    # LiDAR motion: sim is median bin difference in mm (resting baseline is 1-10 mm).
                    # If sim >= 20.0 mm, geometric scan displacement occurred -> platform is moving,
                    # releasing stationary mode promptly without dragging on the resting anchor.
                    is_lidar_static = (sim is None or sim < 20.0)
                    is_static_sample = is_lidar_static and is_imu_static

                    print(
                        f"Motion check: is_static={is_static_sample} "
                        f"(sim={sim if sim is not None else 0.0:.2f}, max_gyro={gyro_max:.4f} rad/s, accel_var={accel_var:.5f})"
                    )

                    if is_static_sample:
                        static_count += 1
                    else:
                        static_count = 0
                prev_scan_bins = current_bins

                is_platform_static = (static_count >= ZUPT_CONSECUTIVE_REQUIRED)
                filter.is_static = is_platform_static

                # Adapt residual biases only when confirmed stationary for a sustained dwell time (static_count >= 10, ~0.7s)
                # using mean gyro over window (not noisy peak max_gyro), strict LiDAR scan similarity, and tight slew rate limits.
                if is_platform_static and static_count >= 10 and len(recent_imu) >= 5:
                    mean_accel = np.mean([m[2] for m in recent_imu], axis=0)
                    mean_gyro = np.mean([m[1] for m in recent_imu], axis=0)
                    mean_gyro_norm = float(np.linalg.norm(mean_gyro))
                    # Strict validation: LiDAR stillness confirms zero motion (sim < 25mm), vibration is quiet (accel_var < 0.08),
                    # and mean rotation is small (< 0.08 rad/s, ~4.6 deg/s) ensuring slow real turns are never learned as bias.
                    if (sim is None or sim < 25.0) and accel_var < 0.08 and mean_gyro_norm < 0.08:
                        step_ba = np.clip(0.01 * (mean_accel - filter.state.ba), -0.003, 0.003)
                        step_bg = np.clip(0.01 * (mean_gyro - filter.state.bg), -0.001, 0.001)
                        filter.state.ba += step_ba
                        filter.state.bg += step_bg

                if update_applied:
                    update_count += 1
                    # Merge the correction computed from state_old into the latest
                    # IMU-predicted state; always merge deltas incrementally to preserve fast integration.
                    delta_p = state.p - state_old.p
                    delta_R = state_old.R.T @ state.R
                    if is_platform_static:
                        if static_anchor_pos is None:
                            static_anchor_pos = filter.state.p[:2].copy()
                            static_anchor_R = filter.state.R.copy()
                        # Allow small residual pull towards anchor/map, but strictly clamp
                        # total position deviation from the stationary resting anchor to at most 3 cm (0.03m)
                        p_cand = filter.state.p[:2] + delta_p[:2]
                        dist_from_anchor = float(np.linalg.norm(p_cand - static_anchor_pos))
                        MAX_STATIONARY_DRIFT = 0.03
                        if dist_from_anchor > MAX_STATIONARY_DRIFT:
                            p_cand = static_anchor_pos + (p_cand - static_anchor_pos) / dist_from_anchor * MAX_STATIONARY_DRIFT
                        filter.state.p[:2] = p_cand
                        filter.state.p[2] = 0.0
                        filter.state.v = np.zeros(3)
                        delta_v = np.zeros(3)

                        # Lock heading to resting anchor heading while stationary
                        if static_anchor_R is not None:
                            filter.state.R = static_anchor_R.copy()
                    else:
                        if static_anchor_pos is not None:
                            # Just released from static anchor: reset live pose cleanly to the true anchor
                            # to wipe any sub-3cm stationary creep that accumulated while resting
                            filter.state.p[:2] = static_anchor_pos.copy()
                            static_anchor_pos = None
                            static_anchor_R = None
                            # On transition scan, bound step to plausible physical step limit (<= 20 cm) in body frame
                            delta_p_body = filter.state.R.T @ delta_p
                            delta_p_body[0] = np.clip(delta_p_body[0], -0.20, 0.20)
                            delta_p = filter.state.R @ delta_p_body
                        filter.state.p += delta_p
                        filter.state.p[2] = 0.0

                    if is_platform_static:
                        filter.state.v = np.zeros(3)
                        delta_v = np.zeros(3)
                    else:
                        # Reconcile velocity from successive LiDAR-committed positions to prevent open-loop IMU runaway
                        if last_committed_time is not None and last_committed_pos is not None:
                            dt_scan = scan_timestamp - last_committed_time
                            if 0.08 <= dt_scan <= 1.5:
                                disp = float(np.linalg.norm(filter.state.p[:2] - last_committed_pos[:2]))
                                is_jump = (disp > 0.12 or (disp / max(dt_scan, 0.01)) > 1.2)
                                if is_jump:
                                    # Suppress false linear velocity synthesis on discontinuous ICP jumps (> 0.12m per scan or > 1.2 m/s)
                                    v_obs = np.zeros(3)
                                elif disp < 0.03:  # Under 3cm displacement over scan interval is noise, not real velocity
                                    v_obs = np.zeros(3)
                                else:
                                    v_obs = (filter.state.p - last_committed_pos) / dt_scan
                                    v_obs[2] = 0.0

                                # Non-Holonomic Constraint (NHC):
                                # Ground wheeled platform cannot slide laterally.
                                # Project observed velocity onto body frame and zero out lateral and vertical velocity,
                                # while preserving longitudinal velocity (forward and reverse) along vehicle heading.
                                v_obs_body = filter.state.R.T @ v_obs
                                v_obs_body[1] = 0.0
                                v_obs_body[2] = 0.0

                                # Prevent runaway reverse velocity while preserving legitimate reverse motion
                                MAX_REVERSE_SPEED = 0.5
                                if v_obs_body[0] < -MAX_REVERSE_SPEED:
                                    v_obs_body[0] = -MAX_REVERSE_SPEED

                                v_obs = filter.state.R @ v_obs_body

                                speed_obs = float(np.linalg.norm(v_obs[:2]))
                                MAX_PLATFORM_SPEED = 0.8
                                if speed_obs > MAX_PLATFORM_SPEED:
                                    v_obs[:2] = (v_obs[:2] / speed_obs) * MAX_PLATFORM_SPEED

                                # Pull live IMU velocity towards the LiDAR displacement velocity
                                alpha_v = 0.7
                                v_reconciled = (1.0 - alpha_v) * filter.state.v[:2] + alpha_v * v_obs[:2]
                                delta_v = np.array([
                                    v_reconciled[0] - filter.state.v[0],
                                    v_reconciled[1] - filter.state.v[1],
                                    0.0,
                                ])
                                filter.state.v[:2] = v_reconciled
                                filter.state.v[2] = 0.0
                            else:
                                delta_v = np.zeros(3)
                        else:
                            delta_v = np.zeros(3)

                    last_committed_pos = filter.state.p.copy()
                    last_committed_time = scan_timestamp

                    if not is_platform_static:
                        filter.state.R = filter.state.R @ delta_R
                        filter.state.R = reorthonormalize(filter.state.R)
                    filter.state.bg += state.bg - state_old.bg
                    filter.state.ba += state.ba - state_old.ba
                    filter.state.g[:] = 0.0
                    filter.P = P_new
                    print(
                        f"LiDAR merge committed: delta_p=[{delta_p[0]:.3f}, {delta_p[1]:.3f}], "
                        f"p=[{filter.state.p[0]:.3f}, {filter.state.p[1]:.3f}], "
                        f"v=[{filter.state.v[0]:.3f}, {filter.state.v[1]:.3f}], "
                        f"static={is_platform_static}",
                        flush=True,
                    )
                    if DEBUG_LIDAR:
                        print(
                            "LiDAR merge debug:",
                            f"delta_position={delta_p}",
                            f"delta_velocity={delta_v}",
                            f"position={filter.state.p}",
                            f"bg={filter.state.bg}",
                            f"ba={filter.state.ba}",
                            f"g={filter.state.g}",
                            f"R_error={np.linalg.norm(filter.state.R.T @ filter.state.R - np.eye(3)):.3e}",
                            f"det_R={np.linalg.det(filter.state.R):.12f}",
                            flush=True,
                        )
                # Map point ingestion logic:
                # Add points on forward motion, upon settling from turns, and progressively when rotating into new frontiers.
                current_heading = float(np.arctan2(filter.state.R[1, 0], filter.state.R[0, 0]))
                if last_map_heading is None:
                    heading_diff = 1.0
                else:
                    heading_diff = abs(float(np.arctan2(np.sin(current_heading - last_map_heading), np.cos(current_heading - last_map_heading))))

                pos_diff = float(np.linalg.norm(filter.state.p[:2] - last_map_pos[:2])) if last_map_pos is not None else 1.0

                is_high_speed_turn = (gyro_max is not None and gyro_max > 0.80)
                is_moderate_turn = (gyro_max is not None and gyro_max > 0.20)
                if is_moderate_turn:
                    last_turn_time = time.monotonic()
                turn_settled = (time.monotonic() - last_turn_time >= 0.15)

                can_add_points = False
                if points_world is not None and update_applied:
                    if not is_platform_static and not is_moderate_turn and turn_settled and pos_diff > 0.10:
                        can_add_points = True
                    elif not is_high_speed_turn and heading_diff >= 0.26:
                        # Rotated >= 15 degrees into a new field of view; map frontier points
                        can_add_points = True
                    elif is_platform_static and (heading_diff >= 0.26 or (last_map_pos is None or pos_diff > 0.10)):
                        # Settled at a new heading or position after a maneuver; add one snapshot then quiet
                        can_add_points = True

                if can_add_points:
                    map.add_points(points_world)
                    last_map_pos = filter.state.p.copy()
                    last_map_heading = current_heading

                if is_platform_static:
                    filter.zupt_update()
                    filter.state.v = np.zeros(3)
                    last_committed_pos = filter.state.p.copy()
                    last_committed_time = scan_timestamp
                    print(f"ZUPT applied | static_count={static_count}", flush=True)

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
            with state_lock:
                filter.is_static = True
                filter.state.v[:] = 0.0
            if "closed" in str(e).lower() or isinstance(e, (BrokenPipeError, EOFError)):
                break
            continue

def copy_state(state):
    # deep  copy of the mutable State fields
    return type(state)(
        R=state.R.copy(),
        p=state.p.copy(),
        v=state.v.copy(),
        bg=state.bg.copy(),
        ba=state.ba.copy(),
        g=state.g.copy(),
    )

def state_error(state, state_ref):
    # Builds the 18-dim error-state of `state` relative to `state_ref`, in the
    rot_err = log(state_ref.R.T @ state.R)
    return np.concatenate([
        rot_err,
        state.p - state_ref.p,
        state.v - state_ref.v,
        state.bg - state_ref.bg,
        state.ba - state_ref.ba,
        state.g - state_ref.g,
    ])
