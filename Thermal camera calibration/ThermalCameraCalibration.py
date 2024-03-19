# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 08:05:28 2023

@author: Cheng ZHANG
"""

import os
import cv2
import numpy as np
import sys

def gamma_trans(img, gamma):  # gamma enhancement
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

file_root = "D:/Thermal camera calibration" ###File folder
shape_inner_corner = (6, 5)
size_grid = 0.1
gamma_val = 0.1

Image_root = file_root + '/Thermal images cropped'

w, h = shape_inner_corner
# cp_int: corner point in int form, save the coordinate of corner points in world sapce in 'int' form like (0,0,0), (1,0,0), (2,0,0) ...., (10,7,0)
cp_int = np.zeros((w * h, 3), np.float32)
cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
# cp_world: corner point in world space, save the coordinate of corner points in world space
cp_world = cp_int * size_grid

file_list = os.listdir(Image_root)

points_world = [] # the points in world space
points_pixel = [] # the points in pixel space (relevant to points_world)
i = 0
for img in file_list:
    print('Processing Image ' + str(i))
    i+=1
    Image_path = Image_root + '/' + img
    img_name = img[:img.rindex('.')]
    img = cv2.imread(Image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gray_img_gamma = gamma_trans(gray_img, gamma_val)
    # find the corners, cp_img: corner points in pixel space
    ret, cp_img = cv2.findChessboardCorners(gray_img_gamma, (w, h), None)
    # if ret is True, save
    
    FoundCorners_path = file_root + '/FoundCorner/' + img_name +'.jpg'
    if ret:
        points_world.append(cp_world)
        points_pixel.append(cp_img)
        # view the corners
        cv2.drawChessboardCorners(img, (w, h), cp_img, ret)
        cv2.imwrite(FoundCorners_path,img)

# calibrate the camera
ret, mat_intri, coff_dis, v_rot, v_trans = cv2.calibrateCamera(points_world, points_pixel, gray_img.shape[::-1], None, None)
print ("ret: {}".format(ret))
print ("intrinsic matrix: \n {}".format(mat_intri))
print ("distortion cofficients: \n {}".format(coff_dis))
print ("rotation vectors: \n {}".format(v_rot))
print ("translation vectors: \n {}".format(v_trans))

# calculate the error of reproject
total_error = 0
Reproject_error = []
for i in range(len(points_world)):
    points_pixel_repro, _ = cv2.projectPoints(points_world[i], v_rot[i], v_trans[i], mat_intri, coff_dis)
    error = cv2.norm(points_pixel[i], points_pixel_repro, cv2.NORM_L2) / len(points_pixel_repro)
    Reproject_error.append(error)
    total_error += error
print("Average error of reproject: {}".format(total_error / len(points_world)))

# dedistort and save the dedistortion result
for img in file_list:
    img_name = img[:img.rindex('.')]
    Image_path = file_root + '/Thermal images cropped/' + img
    img = cv2.imread(Image_path)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_intri, coff_dis, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mat_intri, coff_dis, None, newcameramtx)
    
    Dedistort_path = file_root + '/Dedistort/' + img_name +'.jpg'
    cv2.imwrite(Dedistort_path,dst)


###use the calibration result to rectify inspection images

#inspection image folder
Ins_root = "D:/inspection images"
InsImage_root = Ins_root + '/Thermal'
Ins_list = os.listdir(InsImage_root)
for img in Ins_list:
    img_name = img[:img.rindex('.')]
    Image_path = Ins_root + '/Thermal/' + img
    img = cv2.imread(Image_path)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_intri, coff_dis, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mat_intri, coff_dis, None, newcameramtx)
    
    rectified_path = Ins_root + '/Thermal calibrated/' + img_name +'.jpg'
    cv2.imwrite(rectified_path,dst)


