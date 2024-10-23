# I guess we do option 1 since I'm already implementing the data preprocessing lol

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import pickle
import re

NEW_SIZE = 300

abs_path = os.path.dirname(os.path.abspath(__file__))

def createDirectories():
    male_pattern = re.compile(r'^...M')
    img_folder = os.path.join(abs_path, "SOCOFing")
    for img in os.listdir(img_folder):
        if male_pattern.match(img):
            # TO DO: Create respective directories, look more into fingerprint recognition techniques/algorithms/etc
        

def processImages():
    img_folder = os.path.join(abs_path,  "images")
    for img in img_folder:
        img_array = cv2.imread(img)
        img_array = cv2.resize(img_array, (NEW_SIZE, NEW_SIZE))