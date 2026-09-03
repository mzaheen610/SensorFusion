
# def camera_thread(cam, state_lock, filter, map, imu_measurement_buffer, lidar_scan_queue):
#         #Get the camera scan at 10Hz
#         frame = cam.get_frame()
        
#         #find visual map points for the current image frame based on current pose and current lidar scan
#         visual_map_points = map.query_visible_voxels(lidar_scan_queue, filter.state) #visible voxel query

#         #project lidar points to the current camera frame (u,v)
#         R_CI = np.eye(3) 
#         t_CI = np.eye(3)

#         T_CI = np.eye(4) #dummy camera imu extrinsics, real values have to be calibrated later
#         T_CI[:3, :3] = R_CI
#         T_CI[:3, 3] = t_CI

#         projected_points_pixels = project_points_to_frame(visual_map_points, T_GI, T_CI)

#         residual_list =[]

#         #get the 8x8 pixel patch surrounding the current lidar point
#         for point in projected_points_pixels:
#             #get the 8x8 patch surrounding the pixel
#             pixel = point[1]
#             u = pixel[0]
#             v = pixel[1]
#             current_patch = frame[u-4:u+4, v-4:v+4]

#             reference_patch = map.get_reference_patch(point)

#             #if the current point doesnt have a patch add the new patch and continue
#             if reference_patch is None:
#                 #attach the patch to the lidar map point
#                 map.add_visual_patch(point, current_patch)
#                 continue

#             #calculate the photometric residual for each of the points
#             residual = calculate_photometric_error(current_patch, reference_patch)

#             #add the point residual to the total residual
#             residual_list.append(residual)
#         #do the camera based update using the residual and Kalman Gain
#         #
