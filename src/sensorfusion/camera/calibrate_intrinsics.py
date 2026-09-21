import glob
import cv2
import numpy as np

# Define chessboard inner corners (columns, rows)
CHECKERBOARD = (9, 6)
square_size = 0.030  # size in meters (e.g., 25mm)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D points real world coordinates
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

images = glob.glob("image_*.jpg")

for fname in images:
  img = cv2.imread(fname)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

  if ret:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Camera Matrix (Intrinsic Parameters):\n", mtx)
print("Distortion Coefficients:\n", dist)

# Save parameters using NumPy or Pickle
np.savez("calibration_params.npz", mtx=mtx, dist=dist)
