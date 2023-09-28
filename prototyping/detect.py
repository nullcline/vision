import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from random import randint
import timeit
import time
from cmath import acos, cos, pi, sin, sqrt

from math import atan2
from numpy.lib.arraysetops import isin

from sklearn.cluster import KMeans

# Converting lat/long to cartesian
<<<<<<< Updated upstream:prototyping/detect.py
=======
plots = True
pickle_file_name = "points.pickle"
pickle_file_path = path.join(path.dirname(path.dirname(path.abspath(__file__))), pickle_file_name)
print(f"Will save disks to {pickle_file_path}")
if path.isfile(pickle_file_path):
    os.remove(pickle_file_path)

>>>>>>> Stashed changes:detect.py

class DataPoint:

    def __init__(self, id, lat, lon, color, angle):

        self.id = id
        self.lat = lat
        self.lon = lon
        self.color = color
        self.angle = angle

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

class Disk:

    def __init__(self, id, x, y, color):

        self.id = id
        self.x = x
        self.y = y
        self.color = color
        

#def get_cartesian(lat=None, lon=None):
    # lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    # R = 6371  # radius of the earth
    # x = R * np.cos(lat) * np.cos(lon)
    # y = R * np.cos(lat) * np.sin(lon)
    # z = R * np.sin(lat)
    #return lat, lon, 0

def gps2cartesian(lat1, lon1, lat2, lon2, avelat, planeheight):

    #d is the distance in radians

    #distancekm = radiusofearth_km * distance_radians = 6371 * d

    #distancekm = 6371 * d

    #6371 is the average radius of the earth -- need to use texas one where competition is

    #calculated using texas latidude (31.9686 N) and fort worth height above sea level plus plane height = 215 m
    #now km to m conversion and R = √ [ (r1² * cos(B))² + (r2² * sin(B))² ] / [ (r1 * cos(B))² + (r2 * sin(B))² ] this formula; latitude B, radius R, radius at equator r1, radius at pole r2

    #radius for plane in fort worth, texas = 6372.394

    #distancem = distancekm*1000

    #Radians = Degrees * PI / 180

    radiusatlocation = sqrt((((((6378.137**2)*cos(average_latitude))**2) 
    + ((6356.752**2)*sin(average_latitude))**2))
    /(((6378.137*cos(average_latitude))**2) 
    + ((6356.752*sin(average_latitude))**2)))

    print('Radius in m at location =', radiusatlocation)

    differencelong = lon - starting_lon
    
    differencelat = lat - starting_lat

    distx = differencelong*(pi/180)

    disty = differencelat*(pi/180)

    distancemx = radiusatlocation * distx * 1000

    distancemy = radiusatlocation * disty * 1000

    print('The distance between the two points in meters x is', distancemx, 'and in meters y is:', distancemy)

    x = distancemx
    y = distancemy

    return x,y


def generate_video(lat1, lon1, lat2, lon2, avelat, planeheight, n=5):

    # @param lat: initial latitude
    # @param lon: initial longitude
    # @param n:   number of disks to add

    x, y = gps2cartesian(lat1, lon1, lat2, lon2, avelat, planeheight)
    print(x, y)

    # make long grass field
    grass = np.zeros((720*10, 1280, 3), np.uint8)
    h, w, _ = grass.shape
    grass[:] = (25, 255, 25)

    vid = []
    distance = 0
    padding = 20

    for _ in range(n):

        # draw circle at rando position
        color = (randint(0, 255), 0, 255)
        pos = (randint(0+padding, w-padding), randint(0+padding, h-padding))
        grass = cv2.circle(grass, pos, 20, color, -1)

    while (720+distance < 7200):
        vid.append(grass[0 + distance:720 + distance, :])
        distance += 20

    out = cv2.VideoWriter('generated.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280, 720))

    for frame in vid:   
        out.write(frame)

starting_lon = 50
starting_lat = 50

lon = 52
lat = 52

#for radius calculations

#use average latitude of plane location
average_latitude = 31.9686

#use average plane height in m
planeheight_m = 215

x,y = gps2cartesian(starting_lon, starting_lat, lon, lat, average_latitude, planeheight_m)

#starting_lat and starting_lon = (0,0) on x-y plane

print('Therefore, x and y = (',x,',',y, ')')
# 1 meter apart = .00000024515 coordinates apart
# example below

# starting_lat = 20.0000001
# starting_lon = 120.0000001

#lat = 20.00000024515
#lon = 120.00000024515

if __name__ == "__main__":

    # generate_video(n=3)

    # read video
    # cap = cv2.VideoCapture('generated.avi')

    cap = cv2.VideoCapture('test_1.mp4')
    disks = []

    if (cap.isOpened()== False): 
        print("you fucked up")
    
    # setup
    _, frame = cap.read()
    h, w, _ = frame.shape
    start = timeit.default_timer()
    total_frames = 0
    disk_count = 0
    plane_x, plane_y = 0, 0

    # random intial position, we want to increment by 1.3 ft per frame at 30fps
    # the equivalent in meters is about 0.4 m, or in lat/long, about 0.000005, er, units
    lat = 0.0
    lon = 0.0
    
    # start using a cartesian plane, with origin at runway
    x = 500
    y = 0

    # maybe implement an image thread and process it in the main thread instead of this?]
    # https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:

            total_frames += 1
            # masking away the grass, using an erosion kernel to clean up the noise 

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (20, 92, 0), (48, 255, 210))
            inv = cv2.bitwise_not(mask)
            erode = cv2.erode(inv, np.ones((7,7), np.uint8), iterations=1)
            no_grass = cv2.bitwise_and(frame,frame, mask=erode)
    
            # contour detection
            gray = cv2.cvtColor(no_grass, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
            contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1] 

            for c in contours:
                
                if len(c) > 30 and cv2.contourArea(c) < 50000:
                    # compute the center of the contour
                    M = cv2.moments(c)
                    if M["m00"] != 0:

                        # disk confirmed
                        disk_count += 1 
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # line between disk and center of frame
                        # cv2.line(frame, (int(w/2), int(h/2)), (cX+1, cY+1), (0, 0, 255), 1)

                        v1 = np.array([cX-w/2, cY-h/2])
                        v2 = np.array([1, 0])
                        dot = np.dot(v1, v2)
                        det = v1[0]*v2[1] - v1[1]*v2[0]
            
                        # angle between disk and center of frame
                        #angle = np.arctan2(cY - h/2, cX - w/2)
                        angle = atan2(det, dot)

                        # maybe this is hacky but it works
                        if angle < 0:
                            angle += 2*np.pi

                        # cv2.putText(frame, str(round(angle*180/np.pi,3)), (cX, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                        
                        # record colur of disk, the current gps reading, and the angle
                        color = (int(frame[cY, cX][0]), int(frame[cY, cX][1]), int(frame[cY, cX][2]))
                        cv2.rectangle(frame, (0,0), (50,50), tuple(color), -1)

                        disks.append(DataPoint(disk_count, round(x,7), round(y,7), color, round(angle, 4)))
            
            # calculate change in position based on new gps reading
            delta_x = 0
            delta_y = 1

            x += delta_x
            y += delta_y
                    
            cv2.imshow("Debug", frame)
            #cv2.imshow("thresh", no_grass)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
             break
        
    # Break the loop
        else: 
            print(f"FPS: {total_frames/(timeit.default_timer() - start)}")
            cap.release()
            break
    
    estimated_x = []
    estimated_y = []
    colours = []
    intersection_count = 0

    for disk_1 in disks:
        for disk_2 in disks:

            # 3 conditions on if we bother checking the intersection
            # 1. the disks are not the same
            # 2. the disks are not the same colour
            # 3. the disks weren't found close to eachother

            # @TODO: this sucks balls, we should be able to track each disk by its own id
            # and then just iterate through each of the individual id's and check for intersections

        
            if (disk_1 != disk_2
                and disk_1.color == disk_2.color):

            #if disk_1 != disk_2:
                
                x1, y1, _ = gps2cartesian(disk_1.lat, disk_1.lon)
                x2, y2, _ = gps2cartesian(disk_2.lat, disk_2.lon)

                theta1 = disk_1.angle
                theta2 = disk_2.angle

                m1 = -np.tan(theta1)
                m2 = -np.tan(theta2)

                x = ((y2-m2*x2) - (y1-m1*x1)) / (m1-m2)
                y = m1*x + (y1-m1*x1)

                # estimated_points.append(Disk(intersection_count, x, y, disk_1.color))
                # intersection_count += 1
                
                # remove weird outliers..
                # if (abs(x) > 10000 or abs(y) > 10000):
                #     continue

                if abs(x) > 10000:
                    print(f"t1:{theta1} t2:{theta2}")
                    print(f"x1:{x1} y1:{y1} x2:{x2} y2:{y2}")
                    print(f"disk:{disk_1.id} disk2: {disk_2.id}" )

                if (abs(x) > 1000):
                    continue

                estimated_x.append(x)
                estimated_y.append(y)

                # opencv works in BGR, so we need to convert to RGB

                colours.append((disk_1.color[2], disk_1.color[1], disk_1.color[0]))

    # uncomment to add a useless line of dots 

    # for disk in disks:
    #     estimated_x.append(disk.lat)
    #     estimated_y.append(disk.lon)
    #     colours.append((0,0,0))

    plt.scatter(estimated_x, estimated_y, c=np.array(colours)/255.0, marker=".")
    plt.show()

    # @TODO: Run whatever stats on the results to get the final cartesian positions of the disks.
    # Then, convert these back to GPS locations for the PADA.
    
    valid_x = []
    valid_y = []

    # #500 is where the plane is flying
    # for i in range(len(estimated_x)):
    #     if (estimated_x[i] > 500):
    #         valid_x.append(estimated_x[i])
    #         valid_y.append(estimated_y[i])
    # valid_x = np.array(valid_x)
    # valid_y = np.array(valid_y)

    X = np.zeros((len(estimated_x),2))
    X[:,0] = np.array(estimated_x)
    X[:,1] = np.array(estimated_y)

    kmeans = KMeans(len(color), random_state=0).fit(X)
    print(kmeans.cluster_centers_)

    # Closes all the frames
    cv2.destroyAllWindows()
    # generate_video(lon, lat, 10)