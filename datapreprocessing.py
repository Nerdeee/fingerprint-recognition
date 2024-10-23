# I guess we do option 1 since I'm already implementing the data preprocessing lol

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import pickle

NEW_SIZE = 300

img_folder = os.path.join("/",  "images")
for img in img_folder:
    img_array = cv2.imread(img)
    img_array = cv2.resize(img_array, (NEW_SIZE, NEW_SIZE))