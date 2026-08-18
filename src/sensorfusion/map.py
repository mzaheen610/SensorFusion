"""Map implementation for the sensor fusion system."""

from scipy.spatial import KDTree
import numpy as np

class Map:
    """Basic map implementation using KD tree"""
    def __init__(self):
        self.points = np.empty((0, 3))  # Initialize an empty point cloud

        self.tree = None

    def add_points(self, points):
        self.points = np.vstack([self.points, points])

    def build_tree(self):
        self.tree = KDTree(self.points)

    def query(self, point, k=10):
        #Returns the k nearest neighbors of the point in the map
        if self.tree is None:
            return []
        distances, indices = self.tree.query(point, k=k)
        return self.points[indices], distances    