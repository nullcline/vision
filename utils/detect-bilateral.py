import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from random import randint

# Converting lat/long to cartesian


def get_cartesian(lat=None, lon=None):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371  # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z

lat = 27.2046125
lon = 7.4977123

def generate_video(lat, lon, n):

    # @param lat: initial latitude
    # @param lon: initial longitude
    # @param n:   number of disks to add

    x_0, y_0, _ = get_cartesian(lat, lon)
    print(x_0, y_0)

    # make long grass field
    grass = np.zeros((720*10, 1280, 3), np.uint8)
    grass[:] = (80, 170, 20)

    vid = []
    distance = 0

    for _ in range(n):

        # draw circle at rando position
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        pos = (randint(0, 1080), randint(0, 7200))
        grass = cv22.circle(grass, pos, 20, color, -1)

    while (720+distance < 7200):
        vid.append(grass[0 + distance:720 + distance, :])
        distance += 50

    for frame in vid:   
        print("here")
        cv2.imshow("cam", frame)
        cv2.waitKey(1)

    

    # pretend we have a video stream

    # image = cv22.imread("whatever")

    #   thresh = cv22.threshold(blurred, 10, 255, cv22.THRESH_BINARY)[1]
    #   cnts = cv22.findContours(thresh.copy(), cv22.RETR_EXTERNAL, cv22.CHAIN_APPROX_SIMPLE)
    #   cnts = imutils.grab_contours(cnts)

    #   for c in cnts:
    #     # compute the center of the contour
    #     M = cv22.moments(c)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     # draw the contour and center of the shape on the image
    #     cv22.drawContours(img, [c], -1, (0, 255, 0), 2)
    #     cv22.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    #     # show the image
    #     cv22.imshow("cam", img)
    #     cv22.waitKey(0)

if __name__ == "__main__":

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture('test_0.mp4')
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("ur an IDIOT")
    
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        if ret == True:

            bilateral = cv2.bilateralFilter(frame, 6, 175, 175)
            edge = cv2.Canny(bilateral, 0, 255)
                    
            cv2.imshow("cam", edge)
            
            # Press Q on keyboard to  exit
            # Press Q on keyboard to  exit
            if cv2.waitKey(30) & 0xFF == ord('q'):
             break
        
    # Break the loop
        else: 
            break
    
    # When everything done, release 
    # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
    # generate_video(lon, lat, 10)