"""
Asynchronously-sampled LiDAR points are first re-
combined into scans at the camera’s sampling time through
scan recombination.
"""


class LidarScan:
    def __init__(self):
        self.scan = []

    def recombine(points, scan):
    #recombine the lidar points into a single scan at the cameras scan frequency(eg. 10Hz)
        while(points):
            scan.append(points)
        return scan

