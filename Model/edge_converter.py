import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from os import getcwd as pwd

def detect_edge(dir, folders):
    for folder in folders:
        folder_path = os.path.join(dir, folder)
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            img_edge = cv2.Canny(img, 100, 200)
            path = os.path.join("./new_input/images/", folder, filename)
            cv2.imwrite( path, img_edge );

detect_edge("./input/images/", ["train", "test"])
