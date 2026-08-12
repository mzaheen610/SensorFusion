"""
Asynchronously-sampled LiDAR points are first re-
combined into scans at the camera’s sampling time through
scan recombination.
"""

from initialize import Lidar
import time

class LidarScan:
    def __init__(self):
        self.scan = []

    def get_scan(self):
        return self.scan
    
    def start_scan(self):
        #
        lidar = Lidar()
        self.scan.append((time.time(), lidar.get_readings()))

    def stop_scan(self):
        lidar = Lidar()
        lidar.lidar.stop_motor()
        lidar.lidar.stop()

    def recombine(points, scan):
    #recombine the lidar points into a single scan at the cameras scan frequency(eg. 10Hz)
        while(points):
            scan.append(points)
        return scan

