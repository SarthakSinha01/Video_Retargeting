import cv2
import numpy as np


img = cv2.imread('TargetFrames/New7/t_frame1.jpg')
H = img.shape[0]
W = img.shape[1]
fps = 5

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("target_video7.avi", fourcc, fps, (W,H))

for i in range(0,3999):
    frame = cv2.imread('TargetFrames/New7/t_frame%d.jpg' %i)
    #print(frame.shape)
    out.write(frame)
