"""
Visualization for the map using Open3D
"""
import open3d as o3d
import numpy as np
import socket
import struct
import pickle

def receive_stream(robot_ip="10.12.228.214", port=5000):
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
        window_name="SLAM Map",
        width=1280,
        height=720
    )

    pcd = o3d.geometry.PointCloud()
    visualizer.add_geometry(pcd)

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
            map_points = payload["map_points"]
            points = np.asarray(
                map_points,
                dtype=np.float64
            )
            # -----------------------------
            # Check point cloud
            # -----------------------------
            if points.size == 0:
                print("No points received")
                continue
            if points.ndim > 2:
                points = np.vstack(points)
            if points.ndim != 2 or points.shape[1] != 3:
                print("Invalid point shape:", points.shape)
                continue

            print("\nNumber of points:", points.shape[0])
            print("Minimum:", points.min(axis=0))
            print("Maximum:", points.max(axis=0))
            print("Mean:", points.mean(axis=0))
            print("First 10 points:")
            print(points[:10])

            # -----------------------------
            # Update point cloud
            # -----------------------------
            pcd.points = o3d.utility.Vector3dVector(points)
            visualizer.update_geometry(pcd)
            # -----------------------------
            # Set camera on first frame
            # -----------------------------
            if first_frame:
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
    receive_stream(robot_ip="10.12.228.214", port=5000)
