from src.utils.configuration import Configuration
import shutil
import os
import numpy as np
import cv2

def leq(a: float, b: float) -> bool:
    return a <= b

def geq(a: float, b: float) -> bool:
    return a >= b

def pil_to_cv2(pil_image):
    cv2_image = np.array(pil_image)  # Convert to NumPy array
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    return cv2_image

def adjust_coordinate_rectangle(points):
    x_coords, y_coords = zip(*points)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, x_max, y_min, y_max

class FileCleaner():
    def __init__(self):
        self.folder_names = ['maskfolder', 'croppedfolder', 'splittedfolder']

    def clean(self):
        configuration = Configuration()
        for folder_name in self.folder_names:
            folder = configuration.get(folder_name)
            shutil.rmtree(folder)
            os.makedirs(folder)
