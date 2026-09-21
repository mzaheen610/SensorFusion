"""
Visualization for the map using Open3D
"""
import open3d as o3d
import numpy as np
import socket
import struct
import pickle

def receive_stream(robot_ip="10.141.167.214", port=5000):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to Robot at {robot_ip}:{port}...")
    client.connect((robot_ip, port))
    print("Connected! Receiving data...")
    data_buffer = b""
    payload_size = struct.calcsize(">I")
    # -----------------------------------------
    # Create Open3D visualizer ONCE
    # -----------------------------------------
    visualizer = o3d.visualization.Visualizer()

    visualizer.create_window(
        window_name="SLAM Map & Trajectory",
        width=1280,
        height=720
    )

    pcd = o3d.geometry.PointCloud()
    trajectory_line_set = o3d.geometry.LineSet()

    pcd_added = False
    line_set_added = False
    first_frame = True

    try:
        while True:
            # -----------------------------
            # Receive message size
            # -----------------------------
            while len(data_buffer) < payload_size:
                packet = client.recv(4096)
                if not packet:
                    print("Robot disconnected.")
                    return
                data_buffer += packet
            packed_msg_size = data_buffer[:payload_size]
            data_buffer = data_buffer[payload_size:]
            msg_size = struct.unpack(
                ">I",
                packed_msg_size
            )[0]
            # -----------------------------
            # Receive message
            # -----------------------------
            while len(data_buffer) < msg_size:
                packet = client.recv(4096)
                if not packet:
                    print("Robot disconnected.")
                    return
                data_buffer += packet
            frame_data = data_buffer[:msg_size]
            data_buffer = data_buffer[msg_size:]
            # -----------------------------
            # Deserialize
            # -----------------------------
            payload = pickle.loads(frame_data)
            map_points = payload.get("map_points", [])
            raw_colors = payload.get("colors", [])
            raw_trajectory = payload.get("trajectory", [])

            # Safe trajectory extraction (handles numpy array, deque, or tuples)
            if isinstance(raw_trajectory, np.ndarray) and raw_trajectory.ndim == 2:
                trajectory = raw_trajectory.astype(np.float64)
            elif len(raw_trajectory) > 0 and isinstance(raw_trajectory[0], tuple) and len(raw_trajectory[0]) >= 2:
                trajectory = np.array([item[1].p for item in raw_trajectory], dtype=np.float64)
            elif len(raw_trajectory) > 0 and hasattr(raw_trajectory[0], 'p'):
                trajectory = np.array([item.p for item in raw_trajectory], dtype=np.float64)
            else:
                trajectory = np.asarray(raw_trajectory, dtype=np.float64)

            raw_colors = payload.get("colors", [])
            points = np.asarray(
                map_points,
                dtype=np.float64
            )
            colors = np.asarray(raw_colors, dtype=np.float64) / 255.0 

            # -----------------------------
            # Update point cloud
            # -----------------------------
            if points.size > 0:
                if points.ndim > 2:
                    points = np.vstack(points)
                if points.ndim == 2 and points.shape[1] == 3:
                    pcd.points = o3d.utility.Vector3dVector(points)
                    if colors.shape == points.shape:
                        pcd.colors = o3d.utility.Vector3dVector(colors)

                    if not pcd_added:
                        visualizer.add_geometry(pcd)
                        pcd_added = True
                    else:
                        visualizer.update_geometry(pcd)

            # -----------------------------
            # Update trajectory LineSet
            # -----------------------------
            if trajectory.ndim == 2 and trajectory.shape[0] >= 2 and trajectory.shape[1] == 3:
                num_pts = trajectory.shape[0]
                lines = np.column_stack((np.arange(num_pts - 1), np.arange(1, num_pts)))
                line_colors = np.tile([1.0, 0.0, 0.0], (len(lines), 1))  # Bright red trajectory line

                trajectory_line_set.points = o3d.utility.Vector3dVector(trajectory)
                trajectory_line_set.lines = o3d.utility.Vector2iVector(lines)
                trajectory_line_set.colors = o3d.utility.Vector3dVector(line_colors)

                if not line_set_added:
                    visualizer.add_geometry(trajectory_line_set, reset_bounding_box=False)
                    line_set_added = True
                else:
                    visualizer.update_geometry(trajectory_line_set)

            # -----------------------------
            # Set camera on first frame with data
            # -----------------------------
            if first_frame and (pcd_added or line_set_added):
                visualizer.reset_view_point(True)
                first_frame = False

            visualizer.poll_events()
            visualizer.update_renderer()
    except KeyboardInterrupt:
        print("\nStopping stream.")
    except Exception as e:
        print(f"\nStream error: {e}")
    finally:
        visualizer.destroy_window()
        client.close()

if __name__ == "__main__":
    receive_stream(robot_ip="10.141.167.214", port=5000)
