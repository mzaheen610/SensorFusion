import time
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (640, 480)}))
picam2.start()

time.sleep(2) # warm-up
for i in range(20):
    input(f"Press Enter to capture image {i+1}...")
    picam2.capture_file(f"image_{i:02d}.jpg")
    print(f"Captured image_{i:02d}.jpg")

picam2.stop()
