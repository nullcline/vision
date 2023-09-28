import cv2
import numpy as np
import matplotlib.pyplot as plt

# frame = cv2.imread('grass.jpg')
# # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# # mask = cv2.inRange(hsv, (20, 92, 0), (50, 255, 210))
# # cv2.imshow("mask", mask)
# # inv = cv2.bitwise_not(mask)

# # no_grass = cv2.bitwise_and(image,image, mask=inv)
# # #plt.imshow(inv)
# # cv2.imshow("hi", no_grass)
# # cv2.imshow("mask", mask)

# bilateral = cv2.bilateralFilter(frame, 6, 301, 301)
# cv2.imshow("bilateral", bilateral)
# edge = cv2.Canny(bilateral, 150, 200)

# contours = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1] 

# for c in contours:
#     if cv2.isContourConvex(c):
#         # compute the center of the contour
#         M = cv2.moments(c)
#         if M["m00"] != 0:
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#             # draw the contour and center of the shape on the image
#             cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
#             cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
#             cv2.putText(frame, "center", (cX - 20, cY - 20),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
# cv2.imshow("edge", edge)
# cv2.imshow("frame", frame)
# cv2.waitKey(0)

def get_cartesian(lat=None, lon=None):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371*1e6
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)

    # return in
    return x, y, z 

x1, y1, z1 = get_cartesian(lat=0, lon=0)
print(x1, y1, z1)
x2, y2, z2 = get_cartesian(lat=0, lon=0.000005)
print(x2, y2, z2)
print((x1-x2), y1-y2, z1-z2)