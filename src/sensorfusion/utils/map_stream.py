import socket
import struct
import pickle
import time


def tcp_stream_thread(map_ref, filter_ref, state_lock_ref,
                      host='0.0.0.0', port=5000):
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
                # -----------------------------------------
                # Take a snapshot of the map
                # -----------------------------------------
                with state_lock_ref:
                    map_snapshot = []
                    for key, data in map_ref.voxel_map.items():
                        lidar_points = data["lidar"]
                        if lidar_points:
                            map_snapshot.extend(lidar_points)
                    voxel_size = map_ref.voxel_size
                # -----------------------------------------
                # Create payload
                # -----------------------------------------
                payload = {
                    "map_points": map_snapshot,
                    "voxel_size": voxel_size
                }
                # -----------------------------------------
                # Serialize
                # -----------------------------------------
                data = pickle.dumps(
                    payload,
                    protocol=pickle.HIGHEST_PROTOCOL
                )
                # 4-byte message length
                message_size = struct.pack(">I", len(data))
                # -----------------------------------------
                # Send
                # -----------------------------------------
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