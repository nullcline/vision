import cv2
import piexif
import tkinter as tk
from tkinter import filedialog

def main():
    # Open a file dialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("JPEG files", "*.jpg")])

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(extract_metadata(file_path))

def extract_metadata(file_path):
    # Read the image using OpenCV
    image = cv2.imread(file_path)
    # Check if the image is read properly
    if image is None:
        print("Could not read the image. Exiting.")
        return

    # Extract EXIF data using piexif
    exif_dict = piexif.load(file_path)
    lat, lon, alt = (0,0,0,)
    # Extract the GPS information
    if piexif.GPSIFD.GPSLatitude in exif_dict['GPS']:
        lat = convert_to_decimal(exif_dict['GPS'][piexif.GPSIFD.GPSLatitude])
        lon = -1* convert_to_decimal(exif_dict['GPS'][piexif.GPSIFD.GPSLongitude])
    else:
        lat, lon = (-1,-1,)

    # Extract the altitude information
    if piexif.GPSIFD.GPSAltitude in exif_dict['GPS']:
        altitude_numerator, altitude_denominator = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
        alt = altitude_numerator / altitude_denominator
    else:
        alt = -1

    return dict(lat=lat, lon=lon, alt=alt)

def convert_to_decimal(gps_tuple):
    degrees_numerator, degrees_denominator = gps_tuple[0]
    degrees = degrees_numerator / degrees_denominator

    minutes_numerator, minutes_denominator = gps_tuple[1]
    minutes = minutes_numerator / minutes_denominator

    seconds_numerator, seconds_denominator = gps_tuple[2]
    seconds = seconds_numerator / seconds_denominator

    return degrees + (minutes / 60.0) + (seconds / 3600.0)


if __name__ == "__main__":
    main()