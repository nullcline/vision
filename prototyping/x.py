import numpy as np
from numpy.lib.function_base import angle
from math import atan2

gX = 250
gY = 800

# t1
# position we see on screen
cX_1 = 250
cY_1 = 750

####################################################################### 1000 tall
#                                                                     # 1000 wide
#                                                                     # offset by 50 from origin
#              O (250, 750)                                           #
#                                                                     #
#                                  x (500, 500)                       #
#                                                                     #
#                                                                     #
#                                                                     #
#                                                                     #
#######################################################################

# <- true (0, 0)

v_1 = np.array([cX_1-500, cY_1-500])
ref_1 = np.array([1, 0])
dot_1 = np.dot(v_1, ref_1)
det_1 = v_1[0]*ref_1[1] - v_1[1]*ref_1[0]
angle_1 = atan2(det_1, dot_1)
if angle_1 < 0:
    angle_1 += 2*np.pi
print(f"angle: {angle_1} slope: {np.tan(angle_1)}")


# ---------------------------------------------------------------------------
# t2

cX_2 = 250
cY_2 = 250 

####################################################################### 1000 tall
#                                                                     # 1000 wide
#                                                                     # offset by 550 from origin 
#                                                                     # (moved 500 units forward since last frame)
#                                                                     #
#                                  x (500, 500)                       #
#                                                                     #
#               O (250, 250)                                          #
#                                                                     #
#                                                                     #
#######################################################################

v_2 = np.array([cX_2-500, cY_2-500])
ref_2 = np.array([1, 0])
dot_2 = np.dot(v_2, ref_2)
det_2 = v_2[0]*ref_2[1] - v_2[1]*ref_2[0]
angle_2 = atan2(det_2, dot_2)
if angle_2 < 0:
    angle_2 += 2*np.pi
print(f"angle: {angle_2} slope: {np.tan(angle_2)}")

# math
x1 = 500
y1 = 550 # we were offset by 50

x2 = 500
y2 = 1050 # we were offset by 550

m1 = -np.tan(angle_1)
m2 = -np.tan(angle_2)

# m1 = -1
# m2 = 1

x = (((y2-m2*x2) - (y1-m1*x1)) / (m1-m2))
y = m1*x + (y1-m1*x1)

print(x, y)
