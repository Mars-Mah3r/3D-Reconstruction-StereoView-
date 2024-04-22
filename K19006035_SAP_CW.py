import numpy as np
import cv2 as cv
import cv2
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np

#------------------Camera Calibration------------------

images = sorted(glob.glob('/Users/mars/Documents/02-KCL/Year_4/7_SAP/Individual CW/Images/Chessboard/Chessboard??.jpg'))
print(images)

img = cv.imread(images[0]) # Extract the first image as img
print(img.shape)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert to a gray scale image
print(img.shape, gray.shape)
plt.imshow(gray, cmap='gray'); # Visualize gray

retval, corners = cv.findChessboardCorners(image=gray, patternSize=(10,7))
print(corners.shape)
corners = np.squeeze(corners) #Get rid of extraneous singleton dimension
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
img = cv.imread('/Users/mars/Documents/02-KCL/Year_4/7_SAP/Individual CW/Images/Chessboard/Chessboard01.jpg')
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


#------------------------------SIFT-----------------------------------------------

# Directory containing the images
image_directory = '/Users/mars/Documents/02-KCL/Year_4/7_SAP/Individual CW/Images/hammer'

# Function to load images from a directory in sorted order
def load_images(directory):
    images = []
    raw_images = sorted(glob.glob(image_directory + '/hammer??.png'))
    # Using glob.glob to get the list of all png files in sorted order
    for img_path in raw_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # filename = os.path.basename(img_path)
        images.append((img_path, img))
    return images

# Load images using the updated load_images function
images = load_images(image_directory)

# Function to undistort images
def undistort_images(images, camera_matrix, dist_coeffs):
    undistorted_images = []
    for filename, img in images:
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None)
        undistorted_images.append((filename, undistorted_img))
    return undistorted_images

# Load images
images = load_images(image_directory)

# Proceed with SIFT feature extraction on undistorted images
undistorted_images = undistort_images(images, mtx, dist)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# List to store keypoints and descriptors for each image
keypoints_list = []
descriptors_list = []

# Detect keypoints and compute descriptors for each image
for filename, img in undistorted_images:
    keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

# Brute force matcher
bf = cv2.BFMatcher()

# List to store matches for each pair of images
matches_list = []

# Match descriptors across images
for i in range(len(undistorted_images) - 1):
    matches = bf.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)
    good_matches = []
    
    # Apply ratio test
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    matches_list.append(good_matches)

# Visualize matches for each pair of images
for i in range(len(matches_list)):
    img1 = undistorted_images[i][1]  # Use undistorted image
    img2 = undistorted_images[i + 1][1]  # Use undistorted image
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints_list[i], img2, keypoints_list[i + 1], matches_list[i], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'Matches between Image {i+1} and Image {i+2}')
    plt.show()


#---------------------------------3D Reconstruction--------------------------------

all_points_3d = []

def undistort_keypoints(points, camera_matrix, dist_coeffs):
    # Assuming 'points' is a list of (x, y) tuples.
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    undistorted_pts = cv.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted_pts.reshape(-1, 2)

def estimate_pose(undistorted_pts1, undistorted_pts2, mtx):
    E, mask = cv.findEssentialMat(undistorted_pts1, undistorted_pts2, mtx, method=cv.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv.recoverPose(E, undistorted_pts1, undistorted_pts2, mtx)
    return R, t, mask

def triangulate_points(R, t, undistorted_pts1, undistorted_pts2, mtx):
    P0 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    P1 = np.hstack((R, t))
    P0 = mtx @ P0
    P1 = mtx @ P1
    points_4d_hom = cv.triangulatePoints(P0, P1, undistorted_pts1.T, undistorted_pts2.T)
    points_3d = points_4d_hom / points_4d_hom[3]
    points_3d = points_3d[:3].T
    return points_3d

for i in range(len(undistorted_images) - 1):
    print(f"Processing Image Pair {i} and {i+1}")

    # Check if there are enough matches to proceed
    if len(matches_list[i]) >= 5:
        # Prepare and undistort points for the essential matrix calculation
        pts1 = [keypoints_list[i][m.queryIdx].pt for m in matches_list[i]]
        pts2 = [keypoints_list[i+1][m.trainIdx].pt for m in matches_list[i]]
        
        undistorted_pts1 = undistort_keypoints(pts1, mtx, dist)
        undistorted_pts2 = undistort_keypoints(pts2, mtx, dist)

        # Estimate pose between the two images
        R, t, mask = estimate_pose(undistorted_pts1, undistorted_pts2, mtx)

        # Triangulate points to get 3D coordinates
        points_3d = triangulate_points(R, t, undistorted_pts1, undistorted_pts2, mtx)
        all_points_3d.append(points_3d)
        
        #Optionally, visualize the 3D points
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], marker='.')
        ax.set_title(f'3D Reconstruction of Image Pair {i} and {i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    else:
        print(f"Not enough matches were found between image {i} and image {i+1}.")

# Adjusted part for improved 3D Reconstruction with SIFT
all_points_3d = []
for i in range(len(undistorted_images) - 1):
    print(f"Processing Image Pair {i} and {i+1}")

    # Load keypoints and descriptors for the current and next image
    keypoints1, descriptors1 = keypoints_list[i], descriptors_list[i]
    keypoints2, descriptors2 = keypoints_list[i + 1], descriptors_list[i + 1]

    # Find matches using BFMatcher or FLANNMatcher
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Apply ratio test to find good matches
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) >= 5:
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
        
        undistorted_pts1 = undistort_keypoints(pts1, mtx, dist)
        undistorted_pts2 = undistort_keypoints(pts2, mtx, dist)

        R, t, mask = estimate_pose(undistorted_pts1, undistorted_pts2, mtx)

        points_3d = triangulate_points(R, t, undistorted_pts1, undistorted_pts2, mtx)
        all_points_3d.append(points_3d)
        
        # Visualization omitted for brevity
    else:
        print(f"Not enough matches were found between image {i} and image {i+1}.")

# After processing all image pairs
if all_points_3d:
    all_points_3d = np.concatenate(all_points_3d, axis=0)

    # Visualization and further processing of the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_3d)
    # Optionally visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    # Further processing like saving the point cloud or meshing can be done here


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points_3d)

# Optionally, visualize the point cloud
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("output_point_cloud.ply", pcd)

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Apply Poisson reconstruction
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])



# Disparity Map Calculation 
img1 = undistorted_images[0][1]  # First image of the pair
img2 = undistorted_images[1][1]  # Second image of the pair

# Initialize the stereo block matching object
stereo = cv.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=15)

#Compute disparity map
disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0

#Show disparity map

plt.figure(figsize=(10, 5))
plt.imshow(disparity, 'gray')
plt.title('Disparity Map')
plt.colorbar()
plt.show()

# Path to your .ply file
file_path = "/Users/mars/Desktop/hammer.ply"

# Load point cloud data
point_cloud = o3d.io.read_point_cloud(file_path)

# Visualize point cloud
o3d.visualization.draw_geometries([point_cloud])