# Research Papers

This index keeps the paper set aligned with the Notion research workspace and provides a quick map of what each paper contributes to the project.

## Core Papers

- `An_RPLiDAR_based_SLAM_equipped_with_IMU_for_Autonomous_Navigation_of_Wheeled_Mobile_Robot.pdf`
  - Closest match to the current MVP.
  - Supports the LiDAR plus IMU navigation path used by the canonical script.
- `The_Practice_of_Mapping-based_Navigation_System_for_Indoor_Robot_with_RPLIDAR_and_Raspberry_Pi.pdf`
  - Relevant for Raspberry Pi deployment constraints and practical indoor mapping.
  - Useful as a hardware-and-runtime reference for embedded execution.
- `HVL-SLAM_Hybrid_Vision_and_LiDAR_Fusion_for_SLAM.pdf`
  - Relevant to multimodal fusion research.
  - Supports the longer-term camera plus LiDAR direction.
- `LiDAR_Visual_SLAM-Aided_Vehicular_Inertial_Navigation_System_for_GNSS-Denied_Environments.pdf`
  - Relevant to inertial navigation augmentation in GNSS-denied settings.
  - Useful for drift reduction and future localization work.
- `Comparison_of_Visual_and_LiDAR_SLAM_Algorithms_using_NASA_Flight_Test_Data_FINAL.pdf`
  - Useful for algorithm comparison and evaluation framing.
  - Helps justify tradeoffs between visual and LiDAR-centric SLAM approaches.
- `fast_livo.pdf`
  - Relevant as a modern LiDAR-inertial-visual fusion reference.
  - Useful for understanding tighter coupling and high-rate fusion ideas.

## Research Takeaways

- LiDAR plus IMU fusion is the most practical baseline for the current Raspberry Pi implementation.
- Visual and LiDAR fusion is the main multimodal extension path when camera support is added.
- GNSS-denied inertial navigation papers are useful for drift, stability, and long-horizon localization ideas.
- Comparison papers are valuable for evaluating whether the current canonical approach should move toward tighter coupling or a different estimator structure.
