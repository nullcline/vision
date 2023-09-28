import os
import cv2
import pandas as pd
import tkinter as tk
from time import sleep
from tkinter import filedialog
from extract_metadata import extract_metadata
import plotly.express as px

DELAY = 1 #[ms]

def show_image(filepath):
    image = cv2.imread(filepath)
    scaled = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
    cv2.imshow("Image", scaled)
    cv2.waitKey(1)
    return scaled

def plot_data(data):
    fig = px.scatter_mapbox(data, lat="lat", lon="lon", hover_name="alt", hover_data=["alt"],)
    fig.update_layout(mapbox_style="open-street-map")
    fig.show()


def main():
    # Open a directory dialog
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    if not folder_path:
        print("No folder selected. Exiting.")
        return
    # Iterate through all files in the selected directory
    data = []
    for filename in os.listdir(folder_path):
        # Check if the file is a JPEG image
        sleep(DELAY/1000)

        if filename.lower().endswith(".jpg"):
            filepath = os.path.join(folder_path, filename)
            image = show_image(filepath)
            data.append(extract_metadata(filepath))

    df = pd.DataFrame(data)
    plot_data(df)


if __name__ == "__main__":
    main()
