import socket
import struct
import pickle
import time
import numpy as np

def tcp_stream_thread(map_ref, host='0.0.0.0', port=5000):
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
                payload = {
                    "map_points": points,
                    "colors": colors.astype(np.uint8),  # 0-255, keeps payload small
                    }

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