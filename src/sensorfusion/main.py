from forward import ESIKFStateEstimator
from initialize import Lidar, IMU, CameraSensor
import numpy as np
import time
from lidar.backward import backprop
from lidar.update import lidar_thread
from map import Map
from utils.so3_rotation import skew, exp
from utils.projections import project_points_to_frame, project_points_world
from threading import Thread,Lock
from collections import deque
import copy

state_lock = Lock()
buffer_lock = Lock()
imu_measurement_buffer = deque()
imu_state_buffer = deque()
filter = ESIKFStateEstimator()

def imu_thread(imu, filter_ref):
    prev_time = None
    while True:
        now = time.time()
        imu_data = imu.get_readings()
        if imu_data is None:
            continue
        gyro, accel = imu_data
        if gyro is None or accel is None:
            continue

        gyro = np.asarray(gyro, dtype=float)
        accel = np.asarray(accel, dtype=float)

        if (gyro.shape != (3,) or accel.shape != (3,)
            or not np.all(np.isfinite(gyro))
            or not np.all(np.isfinite(accel))
            ):
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
            continue
        prev_time = now
        with state_lock:
            state, cov = filter_ref.predict((gyro, accel), dt)
            imu_state = (now, copy.deepcopy(state)) #store the timestamp and state for backpropogation of LiDAR points
            imu_state_buffer.append(imu_state)


if __name__ == "__main__":
    filter = ESIKFStateEstimator()
    imu = IMU()
    lidar = Lidar()
    cam = CameraSensor()
    map = Map()
    time.sleep(1.0)   # Let BNO055 finish initializing
    filter.state.R, filter.state.bg, filter.state.ba = imu.initialize_rotation_gyro()
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

    lidar_prev_scan_time = {"time": time.time()}

    imu_worker = Thread(target=imu_thread, args=(imu, filter), daemon=True)
    imu_worker.start()

    lidar_worker = Thread(
        target=lidar_thread,
        args=(state_lock, buffer_lock, filter, lidar, map, imu_measurement_buffer, lidar_prev_scan_time),
        daemon=True,
    )
    lidar_worker.start()

    while(True):
        # now = time.time()
        # dt = now - prev
        # prev = now
        # print("IMU prdiction frequency:", (1/dt))

        # """
        # Forward propogation
        # """
        # # imu_data = imu.get_readings()
        # # imu_measurement_buffer.append((now, imu_data[0], imu_data[1])) #store the timestamp and imu readings for backpropogation of LiDAR points
        # # print("gyro: ", imu_data[0])
        # # print("accel: ", imu_data[1])
        # #After getting the IMU readings, we perform the forward propagation of the state using the ESIKF filter

        # state, cov = filter.predict(imu_data, dt)

        # imu_state = (now, state) #store the timestamp and state for backpropogation of LiDAR points
        # imu_state_buffer.append(imu_state)

        # """
        # Backward propogation
        # """
        # #After forward propogation the LiDAR points are backpropogated when the scan arrives
        # scan = lidar.get_readings()
        # if scan is not None:
        #     now = time.time()
        #     lidar_points_compensated = backprop(now, lidar_prev_scan_time, state, scan, imu_measurement_buffer)
        #     lidar_prev_scan_time = now
        #     for i in range(5):
        #         print("Original LiDAR points: ", scan[i])
        #     for i in range(5):
        #         print("Compensated LiDAR points: ", lidar_points_compensated[i])
        #     print("Current position: ", state.p)


        #Get the camera scan at 10Hz
        frame = cam.get_frame()
        
        # #find visual map points for the current frame based on current pose and current lidar scan
        # visual_map_points = map.query_visible_voxels(scan) #visible voxel query

        # #project lidar points to the current camera frame (u,v)
        # R_CI = np.eye(3) 
        # t_CI = np.eye(3)

        # T_CI = np.eye(4) #dummy camera imu extrinsics, real values have to be calibrated later
        # T_CI[:3, :3] = R_CI
        # T_CI[:3, 3] = t_CI

        # projected_points = project_points_to_frame(visual_map_points, T_GI, T_CI)
        # #get the 8x8 pixel patch surrounding the current lidar point
        # #attach the patch to the lidar map point

        # print(frame)
        with state_lock:
            state = filter.state
            print("Current state (x,y,z): ", state.p)


