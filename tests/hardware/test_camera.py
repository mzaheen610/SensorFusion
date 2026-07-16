from picamzero import Camera

cam = Camera()

cam.take_photo("./images/hello")
cam.capture_sequence(f"./images/sequence.jpg", num_images=3, interval=2)
