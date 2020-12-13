import cv2
import matplotlib.pyplot as plt
import pySaliencyMap

img = cv2.imread('Images/box.jpg')

imgsize = img.shape
img_width = imgsize[1]
img_height = imgsize[0]

print(img_width)
print(img_height)
sm = pySaliencyMap.pySaliencyMap(img_width, img_height)

saliency_map = sm.SMGetSM(img)
salient_region = sm.SMGetSalientRegion(img)

#Visualization

plt.subplot(2,2,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Input image')

plt.subplot(2,2,2), plt.imshow(saliency_map, 'gray')
plt.title('Saliency Map')

plt.subplot(2,2,4), plt.imshow(cv2.cvtColor(salient_region, cv2.COLOR_BGR2RGB))
plt.title('Salient region')

plt.show()
print(saliency_map)

cv2.destroyAllWindows()
