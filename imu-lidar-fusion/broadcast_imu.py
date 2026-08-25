import time
import board
import busio
import socket
import json
from adafruit_icm20x import ICM20948

# --- CONFIGURATION ---
# CHANGE THIS TO your laptop's actual IP address!
LAPTOP_IP = "10.41.10.11"  
UDP_PORT = 5005
HZ = 100
# ---------------------

def main():
    # 1. Setup IMU
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = ICM20948(i2c)
    
    # 2. Setup UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print(f"Streaming IMU data to {LAPTOP_IP}:{UDP_PORT} at {HZ}Hz...")
    
    dt = 1.0 / HZ
    
    try:
        while True:
            start_time = time.time()
            
            # Read Sensor
            ax, ay, az = sensor.acceleration
            gx, gy, gz = sensor.gyro
            
            # Pack into a simple dictionary
            data = {
                "ax": ax, "ay": ay, "az": az,
                "gx": gx, "gy": gy, "gz": gz
            }
            
            # Send over Wi-Fi
            sock.sendto(json.dumps(data).encode('utf-8'), (LAPTOP_IP, UDP_PORT))
            
            # Sleep to maintain frequency
            elapsed = time.time() - start_time
            time.sleep(max(0.0, dt - elapsed))
            
    except KeyboardInterrupt:
        print("\nStopping stream.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()