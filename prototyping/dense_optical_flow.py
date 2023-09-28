"""
Author: Kevin Lin
Date: 2022-08-08
Takes in as all the input parameters to convert two frames from an airplane camera pointed down
into an estimated ground speed.

# formula:
# Ground Speed =  FPS * flow_pixels * FOV * π/180 * altitude / (max_pixels * effective_pixel_cover)
https://docs.google.com/document/d/1AaGhp-P0pejZ2Jb3qx1cGxZ8egAyALim2hEADxT46N8/edit?usp=sharing

"""
import cv2
import numpy as np


method = cv2.optflow.calcOpticalFlowSparseToDense  # python refuses to recognize the optflow import so whatever
sample_frequency = 30  # read every sample_frequency pixels for their optical flow instead of all of them


def dense_optical_flow(old_frame, new_frame, fov, fps, altitude, pixels_on_axis, effective_pixel_cover):
    """
    Calculates the ground speed of an airplane camera pointed straight down, given two frames, altitude and other
    constants.

    :param old_frame: previous frame
    :param new_frame: current frame
    :param fov: fov of the camera that captured the flames
    :param fps: fps of the camera feed
    :param altitude: altitude of the camera in meters
    :param pixels_on_axis: how many pixels are on the axis of the path of motion, e.g. 1080 or 1920
    :param effective_pixel_cover: number in the interval (0, 1], 1 if there is no fisheye or less distortion,
           otherwise the proportion of the screen that is not distorted
    :return: the estimated ground speed in meters/second
    """
    assert old_frame.shape == new_frame.shape, "old_frame and new_frame must have the same shape"
    height, width = old_frame.shape[:2]
    old_frame_c = old_frame.copy()
    new_frame_c = new_frame.copy()
    old_frame_c = cv2.cvtColor(old_frame_c, cv2.COLOR_BGR2GRAY)
    new_frame_c = cv2.cvtColor(new_frame_c, cv2.COLOR_BGR2GRAY)
    flow = method(old_frame_c, new_frame_c, None)
    total_x = 0
    total_y = 0
    samples = 0
    for x in range(0, width, 30):
        for y in range(0, height, 30):
            samples += 1
            total_x += flow[y, x, 0]
            total_y += flow[y, x, 1]
    averaged_x = total_x / samples
    averaged_y = total_y / samples
    magnitude = np.sqrt(averaged_x ** 2 + averaged_y ** 2)

    return fps * magnitude * fov * np.pi / 180 * altitude / (pixels_on_axis * effective_pixel_cover)
