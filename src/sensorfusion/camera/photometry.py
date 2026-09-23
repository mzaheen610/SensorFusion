import numpy as np
import time
from queue import Empty
from utils.projections import (
    project_points_to_frame,
    calculate_photometric_error,
    focals,
    dist_coeffs,
    K_cam,
)
from utils.so3_rotation import skew, exp, reorthonormalize
import cv2
import copy

DEBUG_CAMERA = False
def camera_thread(cam, state_lock, buffer_lock, filter, map, imu_state_buffer, camera_scan_queue):

    # --- Rate Tracking Initialization ---
    camera_update_count = 0
    camera_rate_started = time.monotonic()
    camera_rate_last_report = camera_rate_started

    latest_scan = None
    while True:
        try:
            #Get the camera scan at 10Hz
            frame = cam.get_frame()
            frame_timestamp = time.monotonic()
            if frame is None:
                time.sleep(0.01)
                continue

            # Native hardware frame is 640x480; fallback resize only if dimension differs
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (640, 480))

            # Undistort frame using calibrated intrinsics if available
            if dist_coeffs is not None and K_cam is not None:
                frame = cv2.undistort(frame, K_cam, dist_coeffs)

            display_frame = frame.copy()
            #Get the latest compensated lidar scan from the camera queue
            try:
                while True:
                    latest_scan_time, latest_scan = camera_scan_queue.get_nowait() 
            except Empty:
                pass

            if latest_scan is None or latest_scan_time is None or time.monotonic() - latest_scan_time > 0.8:
                if DEBUG_CAMERA:
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
                if DEBUG_CAMERA:
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

            state_time, state_snapshot, P_snap = state_item #state item contains timestamp, state, cov
            state_old = copy.deepcopy(state_snapshot)
            state = copy.deepcopy(state_snapshot)

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

            MIN_GRADIENT_NORM = 25 #Discards flat walls/sensor noise 
            scored_candidates = []
            """
            Optimize the photometric residual calculation and jacobian computation
            """
            #use only the projected points which has considerable strong gradients
            projected_points_filtered = []
            for point in projected_points_pixels:
                pixel = point[1]
                u = int(round(pixel[0]))
                v = int(round(pixel[1]))

                if (v - 4 < 0 or v + 4 > frame.shape[0]
                        or u - 4 < 0 or u + 4 > frame.shape[1]):
                    continue

                Ix_patch = Ix[v-4:v+4, u-4:u+4]
                Iy_patch = Iy[v-4:v+4, u-4:u+4]

                patch_norm = np.linalg.norm(Ix_patch) + np.linalg.norm(Iy_patch)
                if patch_norm > MIN_GRADIENT_NORM and len(projected_points_filtered)<=30:
                    scored_candidates.append((patch_norm, point))
            #get the strongest 30 points based on score for further processing
            scored_candidates.sort(key=lambda x: x[0], reverse=True) #sort descending by contrast score
            projected_points_filtered = [item[1] for item in scored_candidates[:30]]

            #get the 8x8 pixel patch surrounding the current lidar point
            for point in projected_points_filtered:
                #get the 8x8 patch surrounding the pixel
                pixel = point[1]
                u = int(round(pixel[0]))
                v = int(round(pixel[1]))

                if (v - 4 < 0 or v + 4 > frame.shape[0]
                        or u - 4 < 0 or u + 4 > frame.shape[1]):
                    continue

                if DEBUG_CAMERA:
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
            if DEBUG_CAMERA:
                cv2.imshow("Camera View (Lidar Projected)", display_frame)
                cv2.waitKey(1)

            #do the camera based update using the residual and Kalman Gain
            # with state_lock:
            #     P_copy = filter.P.copy()
            #     # state = filter.state.copy()

            try:
                P_inv = np.linalg.inv(P_snap)
            except np.linalg.LinAlgError:
                P_inv = np.linalg.pinv(P_snap)
            sigma_camera = 10.0

            r = np.asarray(residual_list, dtype=np.float64).ravel()
            H = np.asarray(H_rows, dtype=np.float64).reshape(len(r), 18)

            # Compute information matrices directly (Avoids massive NxN matrices)
            weight = 1.0 / (sigma_camera**2)
            S_inv = P_inv + weight * (H.T @ H)
            b = -weight * (H.T @ r)

            # Solve for error state dx
            dx = np.linalg.solve(S_inv, b)

            # Compute Kalman gain (K = A^-1 * H^T * R^-1) for covariance update
            K = np.linalg.solve(S_inv, weight * H.T)

            if DEBUG_CAMERA:
                print("Camera correction candidate:")
                print("rot:", dx[:3])
                print("pos:", dx[3:6])

            max_rotation_correction = np.deg2rad(15.0)
            max_position_correction = 1.0 #1 meter
            max_velocity_correction = 0.3 #m/s tune to the platforms max vel
            if (
                not np.all(np.isfinite(dx))
                or np.linalg.norm(dx[0:3]) > max_rotation_correction
                or np.linalg.norm(dx[3:6]) > max_position_correction
                or np.linalg.norm(dx[6:9]) > max_velocity_correction
            ):
                if DEBUG_CAMERA:
                    print(
                        "Rejecting implausible correction: "
                        f"rotation={np.linalg.norm(dx[0:3]):.3f} "
                        f"position={np.linalg.norm(dx[3:6]):.3f}"
                        f"velocity={np.linalg.norm(dx[6:9]):.3f}"
                    )
                continue
            
            dx[5]     = 0.0  # Zero out Z translation (camera cannot observe vertical heave on planar points)
            dx[6:9]   = 0.0  # Zero out velocity (camera cannot observe velocity directly)
            dx[9:12]  = 0.0  # Zero out gyro bias
            dx[12:15] = 0.0  # Zero out accel bias (STOPS ACCEL BIAS RUNAWAY!)
            dx[15:18] = 0.0  # Zero out gravity

            theta_rot = dx[0:3]
            state.R = state.R @ exp(theta_rot)
            state.R = reorthonormalize(state.R)   # normalize the R matrix to prevent 
            state.p  += dx[3:6]
            state.v  += dx[6:9]
            state.bg += dx[9:12]
            state.ba += dx[12:15]
            state.g  += dx[15:18]
            I_KH = np.eye(P_snap.shape[0]) - K @ H
            P_new = I_KH @ P_snap @ I_KH.T + (sigma_camera**2) * (K @ K.T)

            with state_lock:
                delta_p = state.p - state_old.p
                delta_v = state.v - state_old.v
                delta_R = state_old.R.T @ state.R
                filter.state.p += delta_p
                filter.state.v += delta_v
                filter.state.R = filter.state.R @ delta_R
                filter.state.bg += state.bg - state_old.bg
                filter.state.ba += state.ba - state_old.ba
                filter.state.g += state.g - state_old.g
                # Update only the pose covariance block; do not overwrite velocity or bias covariances with stale P_snap
                filter.P[0:6, 0:6] = P_new[0:6, 0:6]

            # --- RATE TRACKING CALCULATION ---
            camera_update_count += 1
            now = time.monotonic()
            if now - camera_rate_last_report >= 1.0:
                elapsed = now - camera_rate_started
                rate = camera_update_count / elapsed if elapsed > 0 else 0.0
                print(f"Camera update rate: {rate:.1f} Hz | Photometric residuals evaluated: {len(r)}")
                camera_rate_last_report = now

            time.sleep(0.01)

        except Exception as e:
            print(f"[camera_thread] Error: {type(e).__name__}: {e}", flush=True)
            time.sleep(0.02)
            continue

def projection_and_pose_jacobian(T_GI, T_CI, P_G):
    fx = focals[0]
    fy = focals[1]

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
