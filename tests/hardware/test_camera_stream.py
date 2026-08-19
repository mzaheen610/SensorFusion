from picamzero import Camera
import time

cam = Camera()

cam.start_preview()

N = 30
start = time.perf_counter()

for i in range(N):
    frame = cam.capture_array()

end = time.perf_counter()

cam.stop_preview()

print(f"Average FPS: {N / (end - start):.2f}")
print("Shape:", frame.shape)