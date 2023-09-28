from cv2 import cv2
import numpy as np
import time
import math

print("Running k means color clustering algorithm...")

frame = cv2.imread("test_2.jpeg")
rgb_values = frame.reshape((-1,3));
rgb_values = np.float32(rgb_values)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, (centers) = cv2.kmeans(rgb_values, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
labels = labels.flatten()
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(frame.shape)

masked_image = np.copy(frame)
masked_image = masked_image.reshape((-1,3))
cluster = 0
minDistance = 255
i = 0

rgb_color = (19,121,245)
for color in centers:
    eucDistance = math.sqrt((color[0] - 19)**2 + 
                            (color[1] - 121)**2 + 
                            (color[2] - 245)**2)
    if(eucDistance < minDistance):
        cluster = i
        minDistance = eucDistance
    i = i + 1
    print(eucDistance)
    print(color)

cluster = 0
masked_image[labels != cluster] = [0,0,0]
masked_image[labels == cluster] = centers[cluster]
masked_image = masked_image.reshape(frame.shape)

cv2.imshow("Image", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
