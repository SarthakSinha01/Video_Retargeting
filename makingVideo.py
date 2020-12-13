import cv2

img = cv2.imread('OriginalFrames/New7/frame0.jpg')
H = img.shape[0]
W = img.shape[1]

fps = 20

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("Output7.avi",fourcc,fps,(W,H))

for i in range(0,80):
    frame = cv2.imread('OriginalFrames/New7/frame%d.jpg' %i)
    out.write(frame)
