"""Map implementation for the sensor fusion system."""

from scipy.spatial import KDTree
import numpy as np

# class Map:
#     """Basic map implementation using KD tree"""
#     def __init__(self):
#         self.points = np.empty((0, 3))  # Initialize an empty point cloud

#         self.tree = None

#     def add_points(self, points):
#         self.points = np.vstack([self.points, points])

#     def build_tree(self):
#         self.tree = KDTree(self.points)

#     def query(self, point, k=10):
#         #Returns the k nearest neighbors of the point in the map
#         if self.tree is None:
#             return []
#         distances, indices = self.tree.query(point, k=k)
#         return self.points[indices], distances    

"""
Implementing a simple python dict based voxel hash map for the LiDAR + Camera fusion system
"""
class Map:
    def __init__(self):
        self.voxel_map = {}
        self.voxel_size = 0.5

    def add_points(self, points):
        #add lidar points to the voxel map
        for point in points:
            key = self.get_voxel_key(point)
            if key not in self.voxel_map:
                self.voxel_map[key]= {
                    "lidar": [],
                    "image": []
                }
            else:
                self.voxel_map[key]["lidar"].append(point)
    
    def query(self, point):
        #find the nearest neighbors of the given lidar point from the global map
        #assuming planar neighbors will be found in the same voxel
        key = self.get_voxel_key(point)
        neighbors = self.voxel_map[key]["lidar"]
        return neighbors
    
    def get_voxel_key(self, point):
        #get the root voxel key since the voxel is 0.5x0.5x0.5 cube and multiple points could belong to the same voxel
        return tuple(np.floor(point/self.voxel_size))
    
    def query_visible_voxels(self, points):
        #find the voxels in the map nearest to the measured points
        visual_map_points = []
        for point in points:
            voxel_points = self.voxel_map.query(point)
            visual_map_points.append(voxel_points)
        return visual_map_points
    
    def add_visual_patch(self, point, patch):
        #add/attach the visual patch to the lidar point in the global map
        key = self.get_voxel_key(point)
        self.voxel_map[key]["image"].append(patch)

    def get_reference_patch(self, point):
        #choose one image patch as the reference for now
        #will need to find the best patch for reference after score calculation(viewing angle, similarity based) later
        key = self.get_voxel_key(point)
        patch_list = self.voxel_map[key]["image"]
        return patch_list[-1]
