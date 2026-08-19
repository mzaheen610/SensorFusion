from picamzero import Camera
import time
cam = Camera()
cam.still_size = (640, 480)

start = time.perf_counter()
#prev = time.time()
N = 30

for i in range(30):
   # dt = time.time() - prev
   # prev = time.time()
   # freq = 1 / dt
   # print("Camera freq is(Hz):", freq)
    try:
        frame = cam.capture_array()
       # print("Camera data:", frame)
       # print(frame.shape)
        # cam.take_photo("./images")
        # cam.capture_sequence(f"./images/sequence.jpg", num_images=3, interval=2)
    except TimeoutError as e:
        print(f"Error: {e}")
        print("Camera may not be available or not responding")
    except Exception as e:
        print(f"Camera error: {e}")

end = time.perf_counter()
fps = N / (end - start)
print(f"Average camera FPS: {fps:.2f}")
print("Resolution:", frame.shape)