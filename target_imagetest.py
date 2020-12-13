import cv2
import numpy as np
import pyTargetImage
import pyARAP
import pyTargetFrame

img = cv2.imread('Images/tree.jpg')
frame1 = cv2.imread('OriginalFrames/frame41.jpg')
frame2 = cv2.imread('OriginalFrames/frame51.jpg')

t_obj = pyTargetImage.pyTargetImage(img,2,2)
#t_img = t_obj.target_gridImage(2,2)
t_img = t_obj.target_image()
f_obj = pyTargetFrame.pyTargetFrame(frame1, frame2, 2, 2)
f_frame = f_obj.target_frame()

print(frame2.shape)
print(f_frame.shape)

#obj2 = pyARAP.pyARAP(img)
#gs = obj2.get_gridSM()
#print("##################")
#print(gs)

print("******************")
cv2.imshow("TI", t_img)
cv2.imshow("TF", f_frame)
cv2.waitKey(0)
