# Research Summary

This document mirrors the key Notion findings so the repository has enough context to be understandable without opening the research workspace.

## Project Direction

The Embedded Application Lab topic is an embedded LiDAR, IMU, and camera fusion system for improving navigation accuracy on resource-constrained platforms. The current implementation is focused on a Raspberry Pi MVP using 2D LiDAR plus IMU fusion for mapping in GPS-denied or indoor environments.

The broader research direction includes:

- Kalman filtering and Extended Kalman Filtering for sensor fusion.
- LiDAR SLAM and visual/LiDAR SLAM for localization and mapping.
- IMU-aided navigation in GNSS-denied environments.
- Future path planning work such as A* or reinforcement-learning based planning.

## Literature Themes

The Notion research notes identify these recurring themes:

- Hybrid visual and LiDAR SLAM can improve feature uncertainty and mapping quality, especially when visual features and geometry complement each other.
- LiDAR plus IMU SLAM is a practical baseline for omnidirectional mapping and wheeled robot navigation in GPS-denied environments.
- Visual/LiDAR SLAM aided inertial navigation is relevant for longer-term navigation stability where standalone inertial dead reckoning drifts.

## Key Papers

The project paper set currently centers on:

- `An_RPLiDAR_based_SLAM_equipped_with_IMU_for_Autonomous_Navigation_of_Wheeled_Mobile_Robot.pdf`
- `The_Practice_of_Mapping-based_Navigation_System_for_Indoor_Robot_with_RPLIDAR_and_Raspberry_Pi.pdf`
- `HVL-SLAM_Hybrid_Vision_and_LiDAR_Fusion_for_SLAM.pdf`
- `LiDAR_Visual_SLAM-Aided_Vehicular_Inertial_Navigation_System_for_GNSS-Denied_Environments.pdf`
- `Comparison_of_Visual_and_LiDAR_SLAM_Algorithms_using_NASA_Flight_Test_Data_FINAL.pdf`
- `fast_livo.pdf`

See `docs/research-papers.md` for the structured paper index.

## Raspberry Pi Implementation Status

The project has reached a functional MVP stage:

- A real-time LiDAR plus IMU fusion loop exists.
- Headless execution is supported using Matplotlib `Agg`.
- LiDAR acquisition includes reconnect handling and the RPLidar health-check monkey patch.
- IMU acquisition is threaded and feeds the estimator.
- ICP scan matching provides pose correction.
- Occupancy grid mapping produces persistent map images.
- Hardware smoke tests exist for LiDAR, IMU, and camera.

## ROS2 Notes From Notion

Separate Notion notes describe a ROS2 architecture using:

- RPLidar publishing `/scan`.
- RF2O laser odometry publishing `/odom_rf2o`.
- A UDP IMU bridge publishing `/imu/data`.
- `robot_localization` EKF combining laser odometry and IMU data.
- SLAM Toolbox publishing map corrections.

The intended TF chain is `map -> odom -> base_link`, with static sensor transforms from `base_link` to `laser` and `imu_link`.

An important caveat from the ROS2 notes is that the IMU bridge marks orientation as unknown, so EKF settings that trust IMU yaw need either an orientation estimate or adjusted configuration.

## Next Technical Gaps

- Standardize one canonical implementation and avoid accidental regressions from older variants.
- Add run metrics such as scan rate, ICP fitness, EKF innovation, and save latency.
- Add quantitative evaluation scripts for repeatability, drift, and map quality.
- Improve long-run stability with better bias handling, global correction, or loop closure.
- Continue separating prototype code into maintainable modules once the behavior is stable.
