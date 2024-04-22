# Project: StereoView 3D Reconstruction from a set of 2D images

## Overview
This project explores the integration of advanced computer vision and robotic technologies to achieve precise 3D reconstruction for robotic manipulation. The primary objective is to develop a robust system capable of reconstructing three-dimensional objects from two-dimensional images, thereby enabling robots to interact effectively with their environment. By employing techniques such as **camera calibration**, **feature detection**, **pose estimation**, and **dense reconstruction**.

## Features
- **Camera Calibration and Image Capture**: Utilizing [OpenCV](https://opencv.org/) for high-accuracy calibration and image capture to ensure detailed and precise input data for reconstruction.
- **Feature Detection and Matching Process**: Implementation of SIFT (Scale-Invariant Feature Transform) for reliable detection and matching of features across multiple images.
- **Pose Estimation and Triangulation**: Advanced algorithms like RANSAC are used to estimate the pose of the camera and triangulate points in 3D space.
- **Dense Reconstruction**: Using [Open3D](http://www.open3d.org/) to create detailed 3D point clouds and mesh reconstructions from the processed image data.

## Visualizations
- **Camera Setup and Initial Calibration**  
  ![image](https://github.com/Mars-Mah3r/3D-Reconstruction-SteroView-/assets/108829389/d06a5ca4-e02d-4452-968c-cddf9ba33605)
  ![image](https://github.com/Mars-Mah3r/3D-Reconstruction-SteroView-/assets/108829389/9ae50778-33ae-4a3f-bf70-999b11f07d6a)
  ![image](https://github.com/Mars-Mah3r/3D-Reconstruction-SteroView-/assets/108829389/e68326b3-d90a-4240-9e1b-940a58857ff2)

- **Feature Detection and Matching Process**  
<img width="82" alt="image" src="https://github.com/Mars-Mah3r/3D-Reconstruction-SteroView-/assets/108829389/7903b124-949f-47b8-ac71-c8a8ce3bb307">
<img width="71" alt="image" src="https://github.com/Mars-Mah3r/3D-Reconstruction-SteroView-/assets/108829389/67540b74-f4b2-40bd-b23a-94e13b36b973">

![image](https://github.com/Mars-Mah3r/3D-Reconstruction-StereoView-/assets/108829389/eee939b6-8dde-430f-bbdd-b5aa0533dffc)


![image](https://github.com/Mars-Mah3r/3D-Reconstruction-StereoView-/assets/108829389/88cec1ba-d4b9-4aec-bbaf-6da381aa312c)



  
- **3D Reconstruction Output**  
![image](https://github.com/Mars-Mah3r/3D-Reconstruction-SteroView-/assets/108829389/d1ecb533-f4b3-4907-8ed8-5c28e4983604)
![image](https://github.com/Mars-Mah3r/3D-Reconstruction-SteroView-/assets/108829389/5664cf0c-4f60-4f5d-b072-4843811a5b4c)
![image](https://github.com/Mars-Mah3r/3D-Reconstruction-SteroView-/assets/108829389/75162d2c-8bd6-4e85-9e0f-f03fed5acfa4)


## Installation

Use the package manager [pip3](https://pip.pypa.io/en/stable/) to install the following packages.

```bash
pip3 install numpy
pip3 install opencv-python
pip3 install matplotlib
pip3 install glob2
pip3 install mpl-tools
pip3 install scipy
pip3 install open3d 
```

## Usage
Please change the directory to the chessboard and the objects on lines 13, 83, and 114 in the script:
```
images = sorted(glob.glob('/path/to/the/Chessboard??.jpg'))
img = cv.imread('/path/to/the/chessboard01.jpg')
image_directory = ('/path/to/object/images')
```

Then run the file: K19006035_SAP_CW.py.

