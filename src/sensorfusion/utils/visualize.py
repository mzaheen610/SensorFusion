"""
Visualization for the map using Open3D
"""
import open3d as o3d
import numpy as np

def visualize(map: dict):
    #Shows the current state of the world map consisting of LiDAR and Camera points
    #Initialize the point cloud
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    pcd = o3d.geometry.PointCloud() #initialize the point cloud object
    visualizer.add_geometry(pcd)

    while visualizer.poll_events():
        lidar_points = []
        for key in map.keys():
            lidar_points.append(map[key]["lidar"])

        points = np.asarray(lidar_points)
        if points.size > 0:
            if points.ndim >2 :
                points = np.vstack(points)
        pcd.points = o3d.utility.Vector3dVector(points)


        visualizer.update_geometry(pcd)
        visualizer.update_renderer()

    visualizer.destroy_window()