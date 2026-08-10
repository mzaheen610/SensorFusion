"""
Propogate the Lidar scans backward using the IMU poses
to correct motion distortion
"""

def backprop(scan, imu_poses):
    """
    Backward propagate LiDAR points using IMU poses to correct motion distortion
    """
