import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import os.path as path
import os
import pickle
import math
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yolov7.disk_detector import DiskDetector
import tkinter as tk
from tkinter import filedialog
import plotly.express as px
from extract_metadata import extract_metadata
from disk_objects import DetectedDisk, PredictedLandingZone, DiskList, Disk
import pandas as pd

class DEFAULTS:
    average_lat = 31.9686
    average_alt = 215 # Height from sea level [m]
    debug = True
    show_result = True
    thresh = 10
    noise_atten = 50

def x_algorithm(x1, y1, theta1, x2, y2, theta2):
    '''
    Geometry at the core of the X algorithm, written in a separate function cause I tried to make it run faster but failed lol.
    Takes in the relative cartesian position of the camera from the starting point and the angles between the camera and the disks.
    Returns the relative cartesian position of the landing zone from the starting point, to be converted to GPS coordinates.

    A better breakdown of how this works can be seen in the x.py script.
    '''

    m1 = -np.tan(theta1)
    m2 = -np.tan(theta2)
    if m1 == m2:
        return None, None

    x = ((y2 - m2 * x2) - (y1 - m1 * x1)) / (m1 - m2)
    y = m1 * x + (y1 - m1 * x1)

    if x > 10000 or x < -10000:

        if False:
            print("Found outlier")
            print(f"t1:{theta1} t2:{theta2}")
            print(f"x1:{x1} y1:{y1} x2:{x2} y2:{y2}")
            print(f"disk:{detected_disk.id} disk2: {detected_disk.id}")

        return None, None

    return -x, y

class Vision():
    '''
    An object that will be created in the ground telemetry script
    the "analyze" function should be called on every loop and takes in the newest frame and set of flight data, specifically GPS coordinates.
    these values will be converted to a relative plane location in meters (cartesian) and processed into a predicted landing zone.

    These predicted landing zones can be used by get_predcitions() to get a list of landing zones based on kmeans clustering, which represent our
    final predictions. The one with lowest standard deviation should be used as the landing zone we send the PADA to.

    The pipeline is as follows:

    Image --[detect]-->
    List of detected disks (meters relative to start) --[process]-->
    List of predicted locations (GPS locations) --[get_predictions]-->
    Center of cluster of predicted locations ---> PADA
    #@TODO write this out better
    '''

    def __init__(self, lat_0=None, lon_0=None):

        self.lat_0 = lat_0
        self.lon_0 = lon_0

        self.detected_disks = []
        self.landing_zones = []
        # @TODO make this a list of objects

        self.estimated_x = []
        self.estimated_y = []
        self.colours = []
        self.first_detect = True
        self.disk_detector = DiskDetector()
        self.disk_detector.setup()

    def add_datapoint(self, image, lat, lon):
        if self.lat_0 is None:
            self.lat_0 = lat
            self.lon_0 = lon

        # convert the current lat lon into relative cartesian to the starting point
        h, w, _ = image.shape
        x, y    = self.__gps2cartesian(lat, lon)

        # iterate through all detections
        for cX, cY in self.disk_detector.predict(image):
            if cX is None:
                return

            # line between disk and center of frame

            v1 = np.array([cX-w/2, cY-h/2])
            v2 = np.array([1, 0])
            dot = np.dot(v1, v2)
            det = v1[0]*v2[1] - v1[1]*v2[0]

            # angle between disk and center of frame
            #angle = np.arctan2(cY - h/2, cX - w/2)
            angle = math.atan2(det, dot)

            # maybe this is hacky but it works
            if angle < 0:
                angle += 2*np.pi

            # record colur of disk, the current gps reading, and the angle
            colour = "prediction"
            cv2.line(image, (int(w/2), int(h/2)), (cX+1, cY+1), (0, 0, 255), 3)

            detected_disk = DetectedDisk(len(self.detected_disks),
                                            round(x.real, 7),
                                            round(y.real,7),
                                            round(angle, 4),
                                            colour)

            # @TODO: Identify which disk is which
            # This can rely on the colour of the disk and position. We only want to iterate on the same disk for our process loop.

            # x and y are the relative position of the plane, these will be used to calculate the relative positions of the disks
            if self.first_detect:
                self.detected_disks.append(detected_disk)
                self.first_detect = False

                # process this disk
            else:
                self.process_datapoint(detected_disk)
                # self.detected_disks.append(detected_disk)
                self.detected_disks.append(detected_disk)

            if DEFAULTS.debug:
                cv2.imshow('input', cv2.resize(image, (int(1920/2), int(1080/2)), interpolation=cv2.INTER_AREA))
                cv2.waitKey(1)

                # self.plot()


    def process_datapoint(self, detected_disk):
        '''
        Takes current detected disk and runs it against all the ones we've previously recorded in order to generate new predicted points.
        '''

        # run the X algorithm against every recorded disk and add the results to memory
        if len(self.detected_disks) > 2:

            x1, y1, theta1, colour1 = detected_disk.x, detected_disk.y, detected_disk.angle, detected_disk.colour

            for disk in self.detected_disks:
                if colour1 == disk.colour:

                    x2, y2, theta2 = disk.x, disk.y, disk.angle
                    x, y = x_algorithm(x1, y1, theta1, x2, y2, theta2)

                    if x is None:
                        continue

                    lat, lon = self.__cartesian2gps(x, y)

                    landing_zone = PredictedLandingZone(len(self.landing_zones), lat, lon, disk.colour)
                    self.landing_zones.append(landing_zone)

    def get_plot_data(self):
        # Intended for debugging only
        data = []
        for landing_zone in self.landing_zones:
            data.append(landing_zone.__dict__)
            print(landing_zone.__dict__)

        return data



    def get_predictions(self):
        '''
        Uses stats to generate a final list of the landing zones based on the list of predicted landing zones.
        '''

        latitudes = []
        longitudes = []
        colours = []

        print(len(self.landing_zones), "landing zones detected")
        for landing_zone in self.landing_zones:

            latitudes.append(landing_zone.lat)
            longitudes.append(landing_zone.lon)
            colours.append(self.__flip_colour(landing_zone.colour))

        # find the prediction clusters and their distributions
        try:
            X = np.zeros((len(latitudes), 2))
            X[:, 0] = np.array(latitudes)
            X[:, 1] = np.array(longitudes)

            # Determine how many clusters there are
            range_n_clusters = range(2, 11)
            silhouette_avg = []
            for potential_num_clusters in range_n_clusters:
                # initialise kmeans
                kmeans = KMeans(n_clusters=potential_num_clusters, n_init=10)
                kmeans.fit(X)
                cluster_labels = kmeans.labels_

                # silhouette score
                silhouette_avg.append(silhouette_score(X, cluster_labels))

            # Plot the silhouette analysis graph
            if True:
                plt.plot(range_n_clusters, silhouette_avg)
                plt.xlabel("Values of K")
                plt.ylabel("Silhouette score")
                plt.title("Silhouette analysis For Optimal k")
                plt.show()

            optimal_cluster_num = silhouette_avg.index(max(silhouette_avg))

            print(silhouette_avg)
            num_clusters = range_n_clusters[optimal_cluster_num]

            kmeans = KMeans(range_n_clusters[optimal_cluster_num], random_state=0, n_init=10).fit(X)
            print("the number of clusters is " + str(num_clusters))
            cluster_centers = kmeans.cluster_centers_
            cluster_labels = kmeans.labels_
            # print(cluster_labels)

            disk_list = []
            for cluster_index in range(num_clusters):
                # print(f"indexes of points in cluster {cluster_index}: {np.where(cluster_labels==cluster_index)[0]}")
                # get the index in our data (estimated_x or y) that are in certain clusters so we can match the colour
                index = np.where(cluster_labels==cluster_index)[0][0]
                cluster_colour = colours[index]
                cluster_center = cluster_centers[cluster_index]
                disk_list.append(Disk(cluster_index, cluster_center[0], cluster_center[1], cluster_colour))
                print(f"Colour: {cluster_colour}, location: {cluster_centers[cluster_index]}")

        except (ValueError):
            kmeans = KMeans(1, random_state=0).fit(X)
            print("the number of clusters is 1")
            print(kmeans.cluster_centers_)

        # Closes all the frames
        if DEFAULTS.show_result:
            plt.scatter(latitudes, longitudes, c=np.array(colours) / -255.0, marker=".")

            for cluster_index in range(num_clusters):
                plt.scatter(cluster_centers[cluster_index][0], cluster_centers[cluster_index][1], c=[(0,0,0)], marker="x")

            plt.show()

        cv2.destroyAllWindows()

    def __flip_colour(self, colour):
        return (colour[2], colour[1], colour[0])

    def __invert_colour(self, colour):
        return (colour[0], 255- colour[1], colour[2])

    def __gps2cartesian(self, lat, lon, avelat = DEFAULTS.average_lat, alt = DEFAULTS.average_alt):
        '''
        Takes in two GPS positions and calculates the cartesian distances between them

        rough radius for plane in fort worth, texas = 6372.394

        R  = √ [ (r1² * cos(B))² + (r2² * sin(B))² ] / [ (r1 * cos(B))² + (r2 * sin(B))² ]
        B  = Latitude
        r1 = radius at equator
        r2 = radius at pole
        '''

        starting_lat = self.lat_0
        starting_lon = self.lon_0

        # return x, y
        radiusatlocation = math.sqrt((((((6378.137**2)*math.cos(avelat))**2)
                           + ((6356.752**2)*math.sin(avelat))**2))
                           /(((6378.137*math.cos(avelat))**2)
                           + ((6356.752*math.sin(avelat))**2)))

        # print('Radius in m at location =', radiusatlocation)

        x = radiusatlocation * (lon - starting_lon)*(math.pi/180) * 1000
        y = radiusatlocation * (lat - starting_lat)*(math.pi/180) * 1000

        return x, y

    def __cartesian2gps(self, x, y, avelat = DEFAULTS.average_lat, alt = DEFAULTS.average_alt):

        # just the opposite of gps2cartesian

        starting_lat = self.lat_0
        starting_lon = self.lon_0

        radiusatlocation = math.sqrt((((((6378.137**2)*math.cos(avelat))**2)
                            + ((6356.752**2)*math.sin(avelat))**2))
                            /(((6378.137*math.cos(avelat))**2)
                            + ((6356.752*math.sin(avelat))**2)))

        # print('Radius in m at location =', radiusatlocation)

        lon = x / (radiusatlocation*(math.pi/180) * 1000) + starting_lon
        lat = y / (radiusatlocation*(math.pi/180) * 1000) + starting_lat

        return lat, lon

def hex_colour(rgb):
    return '#%02x%02x%02x' % rgb

def main():
    color_map = {'drone': 'black',
                 'truth': 'red',
                 'prediction': 'blue'}

    vision = Vision()
    ground_truth = [{'lat': 49.258777305555554, 'lon': -123.24260088888889, 'type':"truth"},
                    {'lat': 49.258631972222226, 'lon': -123.24228380555556, 'type':"truth"},
                    {'lat': 49.258781472222225, 'lon': -123.24232672222222, 'type':"truth"},
                    {'lat': 49.2586105,         'lon': -123.24265483333333, 'type':"truth"},]

    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    if not folder_path:
        print("idiot")
        return

    rows = []
    for filename in os.listdir(folder_path):

        if filename.lower().endswith(".jpg"):
            filepath = os.path.join(folder_path, filename)
            data = extract_metadata(filepath)
            data["type"] = "drone"
            rows.append(data)
            image = cv2.resize(cv2.imread(filepath), (0,0), fx=0.25, fy=0.25)
            vision.add_datapoint(image, data["lat"], data["lon"])

    rows.extend(vision.get_plot_data())
    rows.extend(ground_truth)

    df = pd.DataFrame(rows)
    df.to_csv("data.csv")
    fig = px.scatter_mapbox(df, lat="lat", lon="lon",
                            color='type', color_discrete_map=color_map,
                            zoom=3.5)
    fig.update_layout(mapbox_style="open-street-map", showlegend=True)

    fig.show()

if __name__ == "__main__":
    main()
