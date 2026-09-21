import os
import cv2
from picamera2 import Picamera2

# Initialize camera in video mode for smooth display
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

output_dir = "calibration_images"
os.makedirs(output_dir, exist_ok=True)

count = 0
total_images = 20

print("\n--------------------------------------------------")
print(" LIVE PREVIEW ACTIVE")
print(" Click the preview window, then:")
print("   - Press [SPACE] to capture an image")
print("   - Press [q] to exit")
print("--------------------------------------------------\n")

while count < total_images:
    # 1. Grab live frame
    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # 2. Add image counter text to preview
    display_frame = frame_bgr.copy()
    cv2.putText(
        display_frame,
        f"Captured: {count}/{total_images} | Press SPACE to capture",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # 3. Show live preview
    cv2.imshow("Camera Live Preview", display_frame)
    
    # 4. Wait for key press (1ms non-blocking)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == 13:  # SPACE or ENTER
        filename = os.path.join(output_dir, f"calib_{count:02d}.jpg")
        cv2.imwrite(filename, frame_bgr)  # save clean frame
        count += 1
        print(f"[{count}/{total_images}] Saved: {filename}")
    elif key == ord('q'):
        print("Exiting.")
        break

cv2.destroyAllWindows()
picam2.stop()
print(f"\nFinished! All images saved in '{output_dir}/'.")