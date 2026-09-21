import cv2
import numpy as np

# Load the saved calibration parameters
calib = np.load("calibration_params.npz")
K = calib["mtx"]
dist = calib["dist"]

# Load a test image
img = cv2.imread("calibration_images/calib_00.jpg")
undistorted = cv2.undistort(img, K, dist)

# Show side-by-side
combined = np.hstack([img, undistorted])
cv2.imwrite("undistort_test.jpg", combined)
print("Saved 'undistort_test.jpg' (Left: Original | Right: Undistorted). Check it out!")