# I guess we do option 1 since I'm already implementing the data preprocessing lol

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import pickle
import re
import shutil

NEW_SIZE = 300

abs_path = os.path.dirname(os.path.abspath(__file__))
re_exceptions = 0
male_dirname = "Male Fingers"
female_dirname = "Female Fingers"

def createDirectories():
    male_pattern = re.compile(r'.*M.*')
    female_pattern = re.compile(r'.*F.*')
    img_folder = os.path.join(abs_path, "SOCOFing")
    for img in os.listdir(img_folder):
        if male_pattern.match(img):
            # TO DO: Split the images in the gender directories into directories for each finger
            if os.path.isdir(male_dirname):
                shutil.copy(img, male_dirname)   
            else:
                os.mkdir(male_dirname)
        elif female_pattern.match(img):
            if os.path.isdir(female_dirname):
                shutil.copy(img, female_dirname)   
            else:
                os.mkdir(female_dirname)
        else:
            re_exceptions += 1

def processImages():
    # TO DO: look more into fingerprint recognition techniques/algorithms/etc, the code below will also need to be tweaked
    img_folder = os.path.join(abs_path,  "images")
    for img in img_folder:
        img_array = cv2.imread(img)
        img_array = cv2.resize(img_array, (NEW_SIZE, NEW_SIZE))


print("Number of images that couldn't be categorized: ", re_exceptions)
