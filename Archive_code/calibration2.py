import numpy as np
import cv2 as cv
import glob

# Termination criteria for the iterative algorithm used in cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
# Adjust the following line for different chessboard sizes
objp = np.zeros((7*10, 3), np.float32)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
obj_points = []  # 3d point in real world space
img_points = []  # 2d points in image plane.

# List of calibration images
images = glob.glob('/Users/mars/Documents/02-KCL/Year_4/7_SAP/Individual CW/Images/Chessboard/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (10, 7), None)

    # If found, add object points, image points (after refining them)
    if ret:
        obj_points.append(objp)

        # Refine the corners and get more accurate results
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (10, 7), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

# Perform camera calibration to return the camera matrix, distortion coefficients, rotation and translation vectors
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
print("Rotation Vectors:\n", rvecs)
print("Translation Vectors:\n", tvecs)

# Calculate the total error of the calibration
total_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(img_points[i], img_points2, cv.NORM_L2)/len(img_points2)
    total_error += error

print("Total error: ", total_error/len(obj_points))
