from forward import ESIKFStateEstimator
from initialize import IMU, CameraSensor
import numpy as np
import time
from lidar.backward import backprop
from lidar.update import lidar_acquisition_process, lidar_thread
from map import Map
from utils.so3_rotation import skew, exp
from utils.projections import project_points_to_frame, project_points_world, calculate_photometric_error
from threading import Thread,Lock
from multiprocessing import Process, Queue
from collections import deque
import copy
# from utils.map_stream import tcp_stream_thread

state_lock = Lock()
buffer_lock = Lock()
imu_measurement_buffer = deque()
imu_state_buffer = deque()
filter = ESIKFStateEstimator()

def imu_thread(imu, filter_ref):
    prev_time = None
    prediction_count = 0
    rate_started = time.monotonic()
    rate_last_report = rate_started
    while True:
        imu_data = imu.get_readings()
        now = time.time()
        #Reject none values and validate the readings 
        if imu_data is None:
            # invalid_reading_count += 1
            continue
        gyro, accel = imu_data
        if gyro is None or accel is None:
            # invalid_reading_count += 1
            continue

        gyro = np.asarray(gyro, dtype=float)
        accel = np.asarray(accel, dtype=float)

        if (gyro.shape != (3,) or accel.shape != (3,)
            or not np.all(np.isfinite(gyro))
            or not np.all(np.isfinite(accel))
            ):
            # invalid_reading_count += 1
            continue

        with buffer_lock:
            #store the timestamp and imu readings for backpropogation of LiDAR points
            imu_measurement_buffer.append((now, gyro.copy(), accel.copy()))
        #Do the forward propogation
        if prev_time is None:
            prev_time = now
            continue
        dt = now - prev_time

        # Reject impossible timing gaps after sensor/serial faults.
        if dt <= 0.0 or dt > 0.1:
            # Resynchronize after a gap; otherwise every subsequent sample is
            # measured against the stale timestamp and is rejected as well.
            prev_time = now
            continue
        prev_time = now

        with state_lock:
            state, cov = filter_ref.predict((gyro, accel), dt)
            imu_state = (now, copy.deepcopy(state)) #store the timestamp and state for backpropogation of LiDAR points
            imu_state_buffer.append(imu_state)

        prediction_count += 1
        report_time = time.monotonic()
        if report_time - rate_last_report >= 1.0:
            elapsed = report_time - rate_started
            rate = prediction_count / elapsed if elapsed > 0 else 0.0
            # This value is the acceleration driving the integration.  At rest it
            # should be close to zero on every axis; otherwise position must drift.
            linear_accel = filter_ref.state.R @ (accel - filter_ref.state.ba) - filter_ref.state.g
            print(
                f"IMU prediction rate: {rate:.1f} Hz | dt: {dt * 1000:.2f} ms | "
                f"linear accel: {linear_accel} | |v|: {np.linalg.norm(state.v):.3f}"
            )
            rate_last_report = report_time


if __name__ == "__main__":
    filter = ESIKFStateEstimator()
    imu = IMU()
    cam = CameraSensor()
    map = Map()

    time.sleep(1.0)   # Let BNO055 finish initializing
    filter.state.R, filter.state.bg, filter.state.ba = imu.initialize_rotation_gyro()
    filter.state.ba = np.zeros(3)
    filter.state.g = np.zeros(3)  #gravity is already removed by the chip's linear_acceleration output; don't subtract it again
    initial_covariance = 100 * np.eye(18)

    print("Initial R:")
    print(filter.state.R)

    u = imu.get_readings()
    a_body = np.array(u[1])
    a_world = filter.state.R @ a_body

    a_body_corrected = a_body - filter.state.ba
    a_world_corrected = filter.state.R @ a_body_corrected
    linear_accel_world = a_world_corrected - filter.state.g

    print("Body accel:", a_body)
    print("Body accel corrected:", a_body_corrected)
    print("World accel corrected:", a_world_corrected)
    print("Linear world acceleration:", linear_accel_world)

    """
    Initial imu residual calculation
    """
    residuals = []
    for _ in range(100):
        reading = imu.get_readings()
        if reading is not None:
            _, accel = reading
            residuals.append(
                filter.state.R @ (accel - filter.state.ba)
                - filter.state.g
            )
        time.sleep(0.01)
    print("Mean stationary residual:", np.mean(residuals, axis=0))
    
    lidar_prev_scan_time = {"time": time.time()}
    # A single-slot queue prevents slow scan processing from allowing serial
    # data to backlog; the acquisition worker always retains the latest scan.
    lidar_scan_queue = Queue(maxsize=1) #queue to store the lidar scans

    # """
    # Start Lidar scan acquisition process
    # """
    # lidar_acquisition_worker = Process(
    #     target=lidar_acquisition_process,
    #     args=("/dev/ttyUSB0", lidar_scan_queue),
    #     daemon=True,
    # )
    # lidar_acquisition_worker.start()

    # """
    # Start the LiDAR processing and update thread
    # """
    # lidar_worker = Thread(
    #     target=lidar_thread,
    #     args=(state_lock, buffer_lock, filter, map, imu_measurement_buffer,
    #           lidar_prev_scan_time, lidar_scan_queue),
    #     daemon=True,
    # )
    # lidar_worker.start()

    time.sleep(10) #wait for the lidar process to initialize properly

    """
    Starting the IMU thread - data acquisition and forward propogation
    """
    imu_worker = Thread(target=imu_thread, args=(imu, filter), daemon=True)
    imu_worker.start()

    # stream_thread = Thread(target=tcp_stream_thread, args=(map, filter, state_lock), daemon=True)
    # stream_thread.start()

    prev_time = time.time()

    while(True):
        now = time.time()
        if now - prev_time >= 1:
            prev_time = now
            with state_lock:
                state = filter.state
                print("Current state (x,y,z): ", state.p)
