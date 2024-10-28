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
    left_patterns = {
        'thumb': re.compile(r'.*Left_thumb.*'),
        'index': re.compile(r'.*Left_index.*'),
        'middle': re.compile(r'.*Left_middle.*'),
        'ring': re.compile(r'.*Left_ring.*'),
        'pinkie': re.compile(r'.*Left_little.*'),
    }
    right_patterns = {
        'thumb': re.compile(r'.*Right_thumb.*'),
        'index': re.compile(r'.*Right_index.*'),
        'middle': re.compile(r'.*Right_middle.*'),
        'ring': re.compile(r'.*Right_ring.*'),
        'pinkie': re.compile(r'.*Right_little.*'),
    }

    img_folder = os.path.join(abs_path, "SOCOFing")

    if not os.path.exists(male_dirname):
        os.mkdir(male_dirname)
    if not os.path.exists(female_dirname):
        os.mkdir(female_dirname)

    for hand in ['Left', 'Right']:
        hand_dirname_male = os.path.join(male_dirname, f"{hand} Hand")
        hand_dirname_female = os.path.join(female_dirname, f"{hand} Hand")
        if not os.path.exists(hand_dirname_male):
            os.mkdir(hand_dirname_male)
        if not os.path.exists(hand_dirname_female):
            os.mkdir(hand_dirname_female)

        for finger in (left_patterns if hand == 'Left' else right_patterns).items():
            finger_dirname_male = os.path.join(hand_dirname_male, finger[0])
            finger_dirname_female = os.path.join(hand_dirname_female, finger[0])
            if not os.path.exists(finger_dirname_male):
                os.mkdir(finger_dirname_male)
            if not os.path.exists(finger_dirname_female):
                os.mkdir(finger_dirname_female)

    for img in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img)
        
        if os.path.isfile(img_path):
            if male_pattern.match(img):
                # Determine the correct finger directory for males
                finger_found = False
                for finger, pattern in male_pattern.items():
                    if pattern.match(img):
                        shutil.copy(img_path, os.path.join(male_dirname, f"{hand} Hand", finger))
                        finger_found = True
                        break
                if not finger_found:
                    shutil.copy(img_path, male_dirname)  # If no specific finger match, place in general male folder
            elif female_pattern.match(img):
                # Determine the correct finger directory for females
                finger_found = False
                for finger, pattern in female_pattern.items():
                    if pattern.match(img):
                        shutil.copy(img_path, os.path.join(female_dirname, f"{hand} Hand", finger))
                        finger_found = True
                        break
                if not finger_found:
                    shutil.copy(img_path, female_dirname)  # If no specific finger match, place in general female folder
            else:
                re_exceptions += 1

def processImages():
    # TO DO: look more into fingerprint recognition techniques/algorithms/etc, the code below will also need to be tweaked
    img_folder = os.path.join(abs_path,  "images")
    for img in img_folder:
        img_array = cv2.imread(img)
        img_array = cv2.resize(img_array, (NEW_SIZE, NEW_SIZE))

createDirectories()

print("Number of images that couldn't be categorized: ", re_exceptions)
