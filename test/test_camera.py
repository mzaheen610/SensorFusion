from picamzero import Camera

cam = Camera()

for i in range(5):
    try:
        frame = cam.capture_array()
        print("Camera data:", frame)
        # cam.take_photo("./images")
        # cam.capture_sequence(f"./images/sequence.jpg", num_images=3, interval=2)
        # signal.alarm(0)  # Cancel alarm

    except TimeoutError as e:
        print(f"Error: {e}")
        print("Camera may not be available or not responding")
    except Exception as e:
        print(f"Camera error: {e}")
