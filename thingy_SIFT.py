import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Directory containing the images
image_directory = '/Users/mars/Documents/02-KCL/Year_4/7_SAP/Individual CW/thingy'

# Function to load images from a directory
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append((filename, img))
    return images

# Load images
images = load_images(image_directory)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# List to store keypoints and descriptors for each image
keypoints_list = []
descriptors_list = []

# Detect keypoints and compute descriptors for each image
for filename, img in images:
    keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

# Brute force matcher
bf = cv2.BFMatcher()

# List to store matches for each pair of images
matches_list = []

# Match descriptors across images
for i in range(len(images) - 1):
    matches = bf.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)
    good_matches = []
    
    # Apply ratio test
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    matches_list.append(good_matches)

# Visualize matches for each pair of images
for i in range(len(matches_list)):
    img1 = images[i][1]
    img2 = images[i + 1][1]
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints_list[i], img2, keypoints_list[i + 1], matches_list[i], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title(f'Matches between Image {i+1} and Image {i+2}')
    plt.show()






