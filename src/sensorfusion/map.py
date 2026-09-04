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
        self.max_points_per_voxel = 20
    def num_points(self):
        count = 0
        for key in self.voxel_map.keys():
            count += len(self.voxel_map[key]["lidar"])
        return count
    
    def is_empty(self):
        if not self.voxel_map:
            return True
        return False
    
    def add_points(self, points):
        #add lidar points to the voxel map
        for point in points:
            key = self.get_voxel_key(point)
            if key not in self.voxel_map:
                self.voxel_map[key]= {
                    "lidar": [],
                    "image": []
                }
            voxel = self.voxel_map[key]
            #cap the points in a voxel to bound the map size
            if len(voxel["lidar"]) < self.max_points_per_voxel:   
                voxel["lidar"].append(point)

            
    def query(self, point, min_points_in_voxel=10, radius_voxels=1):
        # Find neighbors from the current voxel first.
        # Expand to adjacent voxels only when the current voxel is sparse.
        key = self.get_voxel_key(point)
        voxel = self.voxel_map.get(key, None)

        if voxel is None:
            current_points = []
        else:
            current_points = voxel["lidar"]

        if len(current_points) >= min_points_in_voxel:
            pts = current_points[:min_points_in_voxel * 2] #limit the number of points
            return np.array(pts)

        neighbors = list(current_points)
        for dx in range(-radius_voxels, radius_voxels + 1):
            for dy in range(-radius_voxels, radius_voxels + 1):
                for dz in range(-radius_voxels, radius_voxels + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nkey = (key[0] + dx, key[1] + dy, key[2] + dz)
                    nvoxel = self.voxel_map.get(nkey, None)
                    if nvoxel is not None and nvoxel["lidar"]:
                        neighbors.extend(nvoxel["lidar"])
                        if len(neighbors) >= min_points_in_voxel * 2:
                            break  #stop early once we have enough points
        if len(neighbors) == 0:
            return None

        return np.array(neighbors)
    
    def get_voxel_key(self, point):
        #get the root voxel key since the voxel is 0.5x0.5x0.5 cube and multiple points could belong to the same voxel
        return tuple(np.floor(point/self.voxel_size))
    
    def query_visible_voxels(self, lidar_scan_queue, state):
        #find the voxels in the map nearest to the measured points
        #filter based on the camera field of view projected into the lidar FOV
        current_scan = lidar_scan_queue[-1]
        #points should be backpropogated and transformed to the world coordinates
        #search for points within the Camera x Lidar FOV limits
        lidar_range = 12
        theta = np.deg2rad(31) #half of camera horizontal range
        alpha = np.deg2rad(24) #half of camera vertical range

        rcos_theta = lidar_range* np.cos(theta)
        rsin_theta = lidar_range* np.sin(theta)

        rcos_alpha = lidar_range* np.cos(alpha)
        rsin_alpha = lidar_range* np.sin(alpha)

        current_pose = state.p
        x = current_pose[0]
        y = current_pose[1]
        z = current_pose[2]

        visible_points = []
        for point in current_scan:
            if point[0] < x + rsin_theta and point[0] > x - rsin_theta:
                if point[1] < y + rcos_alpha and y - point[1] > 0:
                    if point[2] < z + rsin_alpha and point[1] > z - rsin_alpha:
                        visible_points.append(point)

        visual_map_points = []
        for point in visible_points:
            voxel_points = self.query(point)
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