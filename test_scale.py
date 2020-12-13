import cv2

image = cv2.imread('test.jpg')
r = 200.0/image.shape[1]
dim = (200, int(image.shape[0]*r))

resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("resized", resized)
cv2.waitKey(0)
