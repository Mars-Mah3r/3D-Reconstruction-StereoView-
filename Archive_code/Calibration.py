import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

images = sorted(glob.glob('/Users/mars/Documents/02-KCL/Year_4/7_SAP/Individual CW/Chessboard/Chessboard??.jpg'))
print(images)

img = cv.imread(images[0]) # Extract the first image as img
print(img.shape)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert to a gray scale image
print(img.shape, gray.shape)
plt.imshow(gray, cmap='gray'); # Visualize gray

retval, corners = cv.findChessboardCorners(image=gray, patternSize=(10,7))
print(corners.shape)
corners = np.squeeze(corners) # Get rid of extraneous singleton dimension
print(corners.shape)
print(corners[:5])  #Examine the first few rows of corners
print(retval)

img2 = np.copy(img)  # Make a copy of original img as img2

# Add circles to img2 at each corner identified
for corner in corners:
    coord = (int(corner[0]), int(corner[1]))
    cv.circle(img=img2, center=coord, radius=5, color=(255, 0, 0), thickness=2)

# Produce a figure with the original image img in one subplot and modified image img2 (with the corners added in).
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(img2);


# Refining Corner Locations with `cv.cornerSubPix`

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Set termination criteria as a tuple.
corners_orig = corners.copy()  # Preserve the original corners for comparison after
corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria=criteria) # extract refined corner coordinates.

# Examine how much the corners have shifted (in pixels)
shift = corners - corners_orig
print(shift[:4,:])

img3 = np.copy(img)


obj_grid = np.zeros((10*7,3), np.float32)
obj_grid[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)
print(obj_grid)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(obj_grid[:, 0], obj_grid[:, 1], obj_grid[:, 2])
plt.show()

# Initialize enpty list to accumulate coordinates
obj_points = [] # 3d world coordinates
img_points = [] # 2d image coordinates
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:
    print('Loading {}'.format(fname))
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    retval, corners = cv.findChessboardCorners(gray, (10,7))
    if retval:
        obj_points.append(obj_grid)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        img_points.append(corners2)
retval, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
print(retval) # Objective function value
print(mtx)    # Camera matrix
print(dist)   # Distortion coefficients

# Camera Calibration in OpenCV

#`cv.calibrateCamera`
img = cv.imread('/Users/mars/Documents/02-KCL/Year_4/7_SAP/Individual CW/Chessboard/Chessboard01.jpg')
h,w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(img)
plt.title('Original')
plt.subplot(122)
plt.imshow(dst)
plt.title('Corrected')

total_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(img_points[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(obj_points)
print("Mean reprojection error: ", mean_error)
