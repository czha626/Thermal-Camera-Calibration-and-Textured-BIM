# -*- coding: utf-8 -*-
"""
Created on Mon May  1 13:51:42 2023

@author: Cheng ZHANG
"""

import cv2
import numpy as np
import pandas as pd
import math

def load_img(name: str,mode: int) -> np.ndarray:
    return cv2.imread(name, mode)

file_root = "D:/Image registration"

file_path = file_root+'/'+'CornerPoints.xlsx'
IRimagePoints = pd.read_excel(file_path, sheet_name=1)
IRimagePoints = np.array(IRimagePoints)
IRimagePoints = IRimagePoints.tolist()

RGBimagePoints = pd.read_excel(file_path, sheet_name=0)
RGBimagePoints = np.array(RGBimagePoints)
RGBimagePoints = RGBimagePoints.tolist()

RGB_chess_all = []
IR_chess_all = []
for i in range(30):
    for j in range(15):
        rgb_points = [RGBimagePoints[i][2*j],RGBimagePoints[i][2*j+1]]
        ir_points = [IRimagePoints[i][2*j],IRimagePoints[i][2*j+1]]
        RGB_chess_all.append(rgb_points)
        IR_chess_all.append(ir_points)

four_corner_index = [0, 4, 25, 29]
RGB_chess = []
IR_chess = []
for k in range(4):
    i = four_corner_index[k]
    for j in range(15):
        rgb_points = [RGBimagePoints[i][2*j],RGBimagePoints[i][2*j+1]]
        ir_points = [IRimagePoints[i][2*j],IRimagePoints[i][2*j+1]]
        RGB_chess.append(rgb_points)
        IR_chess.append(ir_points)

        
####RGB to IR
src = np.float32(RGB_chess_all) ## initial image plan
dst = np.float32(IR_chess_all) ## target image plan

####IR to RGB
#src = np.float32(IR_chess_all) ## initial image plan
#dst = np.float32(RGB_chess_all) ## target image plan

IRtoRGBPerspectiveT, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
#IRtoRGBPerspectiveT = cv2.getPerspectiveTransform(src, dst)

###########reproject chess corner point from RGB image to IR image
matrix = IRtoRGBPerspectiveT.tolist()
IR_chess_reproject = []
for i in range(len(RGB_chess_all)):
    x = (RGB_chess_all[i][0]*matrix[0][0] + RGB_chess_all[i][1]*matrix[0][1] + matrix[0][2])/(RGB_chess_all[i][0]*matrix[2][0] + RGB_chess_all[i][1]*matrix[2][1] + matrix[2][2])
    y = (RGB_chess_all[i][0]*matrix[1][0] + RGB_chess_all[i][1]*matrix[1][1] + matrix[1][2])/(RGB_chess_all[i][0]*matrix[2][0] + RGB_chess_all[i][1]*matrix[2][1] + matrix[2][2])
    IR_chess_reproject.append([x,y])

Fusion_error = []
for i in range(len(RGB_chess_all)):
    IR_real = IR_chess_all[i]
    IR_reproject = IR_chess_reproject[i]
    error = math.sqrt((IR_real[0]-IR_reproject[0])**2 + (IR_real[1]-IR_reproject[1])**2)
    Fusion_error.append(error)

np.mean(Fusion_error)



