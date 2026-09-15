import numpy as np
import time
from queue import Empty
from utils.projections import project_points_to_frame, calculate_photometric_error
from utils.so3_rotation import skew, exp
import cv2

DEBUG_CAMERA = True
def camera_thread(cam, state_lock, buffer_lock, filter, map, imu_state_buffer, camera_scan_queue):

    # --- Rate Tracking Initialization ---
    camera_update_count = 0
    camera_rate_started = time.monotonic()
    camera_rate_last_report = camera_rate_started

    latest_scan = None
    while True:
        #Get the camera scan at 10Hz
        frame = cam.get_frame()
        frame_timestamp = time.monotonic()
        if frame is None:
            time.sleep(0.01)
            continue

        # Resize the frame to 640x480 for fast Pi processing and correct math
        frame = cv2.resize(frame, (640, 480))
        
        display_frame = frame.copy()
        #Get the latest compensated lidar scan from the camera queue
        try:
            while True:
                latest_scan_time, latest_scan = camera_scan_queue.get_nowait() 
        except Empty:
            pass

        if latest_scan is None or latest_scan_time is None or time.monotonic() - latest_scan_time > 0.3:
            cv2.imshow("Camera View (Lidar Projected)", display_frame) #show empty frame if lidar data is missing
            cv2.waitKey(1)
            time.sleep(0.01)
            continue

        #Get the latest state near the camera scan timing
        with buffer_lock:
            state_item = next(
                (item for item in reversed(imu_state_buffer) if item[0] <= frame_timestamp),
                None,
            )
        if state_item is None:
            cv2.imshow("Camera View (Lidar Projected)", display_frame)
            cv2.waitKey(1)
            time.sleep(0.01)
            continue

        #convert the RGB image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        #project lidar points to the current camera frame (u,v)
        R_CI = np.array([
            [ 0.0, -1.0,  0.0],  # Camera X (Right) = IMU -Y (Left)
            [ 0.0,  0.0, -1.0],  # Camera Y (Down)  = IMU -Z (Up)
            [ 1.0,  0.0,  0.0]   # Camera Z (Front) = IMU +X (Forward)
        ])
        # R_CI = np.eye(3) 
        T_CI = np.eye(4) #dummy camera imu extrinsics, real values have to be calibrated later
        T_CI[:3, :3] = R_CI
        T_CI[:3, 3] = np.zeros(3)

        _, state_snapshot, _ = state_item #state item contains timestamp, state, cov

        T_GI = np.eye(4)
        T_GI[:3, :3] = state_snapshot.R
        T_GI[:3, 3] = state_snapshot.p

        #find visual map points for the current image frame based on current pose and current lidar scan
        visual_map_points = map.query_visible_voxels(latest_scan, filter.state) #visible voxel query

        projected_points_pixels = project_points_to_frame(
            visual_map_points, T_CI, T_GI
        )
        print(f"DEBUG: Found {len(visual_map_points)} voxels, {len(projected_points_pixels)} projected onto image")

        residual_list =[]
        H_rows = []
        #get the 8x8 pixel patch surrounding the current lidar point
        for point in projected_points_pixels:
            #get the 8x8 patch surrounding the pixel
            pixel = point[1]
            u = int(round(pixel[0]))
            v = int(round(pixel[1]))

            if (v - 4 < 0 or v + 4 > frame.shape[0]
                    or u - 4 < 0 or u + 4 > frame.shape[1]):
                continue

            #Live visualization of the lidar projected pixels on the camera frame
            cv2.circle(display_frame, (u,v), radius=2, color=(0, 255,0), thickness=-1)
            #draw the patch surrounding the pixels
            cv2.rectangle(display_frame, (u-4, v-4), (u+4, v+4), color=(0, 0, 255), thickness=1)
            #add the pixel color to the map point
            color = frame[v,u]
            map.set_point_color(point[0], color)

            #visualize the current frame with the lidar projected points
            current_patch = gray[v-4:v+4, u-4:u+4].astype(np.float32)
            reference_patch = map.get_reference_patch(point[0])

            #if the current point doesnt have a patch add the new patch and continue
            if reference_patch is None:
                #attach the patch to the lidar map point
                map.add_visual_patch(point[0], current_patch)
                continue
            #calculate the photometric residual for each of the points
            residual = calculate_photometric_error(
                current_patch, reference_patch, None
            )

            #build the Jacobian(H) for the camera using the residuals and state
            """
            Jacobian H for the camera model is delta_I (intensity of the pixel) / delta_x
            x --> P_C -->(u,v) --> I(u,v)
            """

            #compute projection jacobian delta_uv/delta_Pc where Pc is the lidar point in camera frame
            #compute pose jacobian delta_Pc/delta_x
            H_pose, _ = projection_and_pose_jacobian(T_GI, T_CI, point[0])
            if H_pose is None:
                continue
            #compute the image gradient delta_I/delta_uv for the current patch
            Ix_patch = Ix[v-4:v+4, u-4:u+4]
            Iy_patch = Iy[v-4:v+4, u-4:u+4]

            for pi in range(Ix_patch.shape[0]):
                for pj in range(Ix_patch.shape[1]):
                    #image gradient jacobian
                    J_image = np.array([Ix_patch[pi, pj], Iy_patch[pi,pj]])
                    #compute the full pixel jacobian delta_I/ delta_x = image_gradient x pose jacobian
                    # 1x2 @ 2x18 = 1x18
                    H_rows.append(J_image @ H_pose)
            #add the point residual to the total residual
            residual_list.extend(np.asarray(residual).ravel())

        #update the Live Window GUI
        cv2.imshow("Camera View (Lidar Projected)", display_frame)
        cv2.waitKey(1)

        #do the camera based update using the residual and Kalman Gain
        with state_lock:
            P_copy = filter.P.copy()
            # state = filter.state.copy()

        P_inv = np.linalg.inv(P_copy)
        sigma_camera = 10.0

        r = np.asarray(residual_list, dtype=np.float64).ravel()
        H = np.asarray(H_rows, dtype=np.float64).reshape(len(r), 18)

        # Compute information matrices directly (Avoids massive NxN matrices)
        weight = 1.0 / (sigma_camera**2)
        S_inv = P_inv + weight * (H.T @ H)
        b = weight * (H.T @ r)

        # Solve for error state dx
        dx = np.linalg.solve(S_inv, b)

        # Compute Kalman gain (K = A^-1 * H^T * R^-1) for covariance update
        K = np.linalg.solve(S_inv, weight * H.T)

        if DEBUG_CAMERA:
            print("Camera correction candidate:")
            print("rot:", dx[:3])
            print("pos:", dx[3:6])

        with state_lock:
            filter.state.R = filter.state.R @ exp(dx[0:3])   
            filter.state.p += dx[3:6]
            filter.state.v += dx[6:9]
            filter.state.bg += dx[9:12]
            filter.state.ba += dx[12:15]
            filter.state.g += dx[15:18]
            
            # Update LIVE covariance so we don't erase IMU predictions made mid-update
            filter.P = (np.eye(18) - K @ H) @ filter.P

        # --- RATE TRACKING CALCULATION ---
        camera_update_count += 1
        now = time.monotonic()
        if now - camera_rate_last_report >= 1.0:
            elapsed = now - camera_rate_started
            rate = camera_update_count / elapsed if elapsed > 0 else 0.0
            print(f"Camera update rate: {rate:.1f} Hz | Photometric residuals evaluated: {len(r)}")
            camera_rate_last_report = now

        time.sleep(0.01)

def projection_and_pose_jacobian(T_GI, T_CI, P_G):
    fx = 529.6
    fy = 528.8

    R_GI = T_GI[:3, :3]
    p_GI = T_GI[:3, 3]
    R_CI = T_CI[:3, :3]
    t_CI = T_CI[:3, 3]
    P_I = R_GI.T @ (P_G - p_GI)
    P_C = R_CI @ P_I + t_CI
    X, Y, Z = P_C
    # Point behind camera
    if Z <= 0:
        return None, None
    
    J_proj = np.array([
        [fx / Z, 0.0, -fx * X / (Z * Z)],
        [0.0, fy / Z, -fy * Y / (Z * Z)]
    ])

    q = P_I
    # dP_C / dp
    J_p = -R_CI @ R_GI.T
    # dP_C / dtheta
    J_theta = R_CI @ skew(q)

    J_PC = np.zeros((3, 18))
    J_PC[:, 0:3] = J_theta
    J_PC[:, 3:6] = J_p
    J_PC[:, 6:9] = 0.0
    J_PC[:, 9:12] = 0.0
    J_PC[:, 12:15] = 0.0
    J_PC[:, 15:18] = 0.0

    H_pose = J_proj @ J_PC # H_pose = Projection jacobian @ Pose Jacobian

    return H_pose, P_C
