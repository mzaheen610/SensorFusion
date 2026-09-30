import socket
import struct
import pickle
import time
import numpy as np

try:
    from camera.photometry import get_latest_debug_frame
except ImportError:
    try:
        from sensorfusion.camera.photometry import get_latest_debug_frame
    except ImportError:
        get_latest_debug_frame = lambda: None

def tcp_stream_thread(map_ref, imu_state_buffer, buffer_lock=None, host='0.0.0.0', port=5000):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[Streamer] Listening on {host}:{port}...")
    while True:
        conn, addr = server.accept()
        print(f"[Streamer] Laptop connected from {addr}")
        try:
            while True:
                points, colors = map_ref.get_all_points_and_colors()

                # Safely extract position coordinates (x, y, z)
                if buffer_lock is not None:
                    with buffer_lock:
                        items = list(imu_state_buffer)
                else:
                    try:
                        items = list(imu_state_buffer)
                    except Exception:
                        items = []

                if items and isinstance(items[0], np.ndarray) and items[0].shape == (3,):
                    traj_pts = [p.copy() for p in items]
                elif items and hasattr(items[0], '__len__') and len(items[0]) >= 2 and hasattr(items[0][1], 'p'):
                    traj_pts = [item[1].p.copy() for item in items]
                elif items and hasattr(items[0], 'p'):
                    traj_pts = [item.p.copy() for item in items]
                else:
                    traj_pts = items

                trajectory = np.asarray(traj_pts, dtype=np.float32) if len(traj_pts) > 0 else np.empty((0, 3), dtype=np.float32)

                # Downsample if buffer is huge (> 5000 points) to keep payload small
                if len(trajectory) > 5000:
                    step = max(1, len(trajectory) // 5000)
                    trajectory = trajectory[::step]

                payload = {
                    "map_points": points,
                    "colors": colors.astype(np.uint8),  # 0-255, keeps payload small
                    "trajectory": trajectory
                }

                cam_frame = get_latest_debug_frame()
                if cam_frame is not None:
                    payload["camera_frame"] = cam_frame

                data = pickle.dumps(
                    payload,
                    protocol=pickle.HIGHEST_PROTOCOL
                )
                # 4-byte message length
                message_size = struct.pack(">I", len(data))

                conn.sendall(message_size)
                conn.sendall(data)
                # 10 Hz
                time.sleep(0.1)
        except (ConnectionResetError, BrokenPipeError):
            print("[Streamer] Laptop disconnected.")

        except Exception as e:
            print(f"[Streamer] Error: {e}")
        finally:
            conn.close()