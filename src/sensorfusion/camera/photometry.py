import numpy as np
import time
from queue import Empty
from utils.projections import project_points_to_frame, calculate_photometric_error

def camera_thread(cam, state_lock, filter, map, imu_measurement_buffer, lidar_scan_queue):
    latest_scan = None
    while True:
        #Get the camera scan at 10Hz
        frame = cam.get_frame()
        
        #find visual map points for the current image frame based on current pose and current lidar scan
        try:
            while True:
                _, latest_scan = lidar_scan_queue.get_nowait()
        except Empty:
            pass
        visual_map_points = map.query_visible_voxels(latest_scan, filter.state) #visible voxel query

        #project lidar points to the current camera frame (u,v)
        R_CI = np.eye(3) 
        t_CI = np.eye(3)

        T_CI = np.eye(4) #dummy camera imu extrinsics, real values have to be calibrated later
        T_CI[:3, :3] = R_CI
        T_CI[:3, 3] = np.zeros(3)

        T_GI = np.eye(4)
        T_GI[:3, :3] = filter.state.R
        T_GI[:3, 3] = filter.state.p

        projected_points_pixels = project_points_to_frame(
            visual_map_points, T_CI, T_GI
        )

        residual_list =[]

        #get the 8x8 pixel patch surrounding the current lidar point
        for point in projected_points_pixels:
            #get the 8x8 patch surrounding the pixel
            pixel = point[1]
            u = int(round(pixel[0]))
            v = int(round(pixel[1]))

            if (v - 4 < 0 or v + 4 > frame.shape[0]
                    or u - 4 < 0 or u + 4 > frame.shape[1]):
                continue

            #add the pixel color to the map point
            color = frame[v,u]
            map.set_point_color(point[0], color)

            #visualize the current frame with the lidar projected points

            current_patch = frame[v-4:v+4, u-4:u+4]

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

            #add the point residual to the total residual
            residual_list.append(residual)

        #do the camera based update using the residual and Kalman Gain
        #
        time.sleep(0.1)
