import os
import glob
import cv2
import numpy as np

# Set to the number of INTERNAL CORNERS (intersections), NOT squares!
# Count inner points along width and height:
CHECKERBOARD = (9, 6) 
square_size = 0.025  # Square size in meters (e.g., 0.025 for 25mm)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Look in calibration_images folder first, then local directory
images = sorted(glob.glob("calibration_images/*.jpg") + glob.glob("image_*.jpg") + glob.glob("calib_*.jpg"))
print(f"Found {len(images)} image files to evaluate.")

if len(images) == 0:
    print("Error: No images found! Check your image folder path.")
    exit(1)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

objpoints = []  # 3D points
imgpoints = []  # 2D points
valid_images = 0
image_size = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    # Find corners with adaptive flags
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # If orientation was transposed, try flipped dimensions
    if not ret:
        alt_board = (CHECKERBOARD[1], CHECKERBOARD[0])
        ret, corners = cv2.findChessboardCorners(
            gray, alt_board,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            # Adjust objp for the alt orientation
            objp_alt = np.zeros((alt_board[0] * alt_board[1], 3), np.float32)
            objp_alt[:, :2] = np.mgrid[0:alt_board[0], 0:alt_board[1]].T.reshape(-1, 2) * square_size
            objpoints.append(objp_alt)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            valid_images += 1
            print(f"  [OK (transposed)] {fname}")
            continue

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        valid_images += 1
        print(f"  [OK] {fname}")
    else:
        print(f"  [FAILED] {fname} - No chessboard detected")

print(f"\nSuccessfully detected corners in {valid_images}/{len(images)} images.")

if valid_images < 5:
    print(f"Error: Need at least 5-10 successful images for calibration (got {valid_images}).")
    print(f"Check that your CHECKERBOARD inner corners tuple ({CHECKERBOARD}) matches your printed sheet!")
    exit(1)

# Run calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

print("\n================ CALIBRATION RESULTS ================")
print(f"Reprojection Error: {ret:.4f} pixels (lower is better, ideally < 0.5)")
print(f"Camera Matrix K (fx, fy, cx, cy):\n{mtx}")
print(f"Distortion Coefficients D (k1, k2, p1, p2, k3):\n{dist.ravel()}")

# Save to disk
np.savez("calibration_params.npz", mtx=mtx, dist=dist)
print("\nSaved parameters to 'calibration_params.npz'")