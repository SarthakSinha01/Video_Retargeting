import cv2
import numpy as np
import pyGenerateTargetFrames

vidObj = cv2.VideoCapture('OriginalVideo/New/test.mp4')
count = 0
success = 1
while success:
    success,image = vidObj.read()
    cv2.imwrite("OriginalFrames/New/frame%d.jpg" % count, image)
    count += 1

th = 2
tw = 2
obj = pyGenerateTargetFrames.GenerateTargetFrames(th, tw, count-1)
obj.genFrames()
