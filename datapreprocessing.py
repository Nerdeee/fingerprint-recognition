# I guess we do option 1 since I'm already implementing the data preprocessing lol

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import pickle
import re
import shutil

abs_path = os.path.dirname(os.path.abspath(__file__))
re_exceptions = 0
male_dirname = "Male Fingers"
female_dirname = "Female Fingers"

def createDirectories():
    # Updated patterns to match exact format
    male_pattern = re.compile(r'.*__M_.*')
    female_pattern = re.compile(r'.*__F_.*')
    
    left_patterns = {
        'thumb': re.compile(r'.*_Left_thumb_finger\.BMP$'),
        'index': re.compile(r'.*_Left_index_finger\.BMP$'),
        'middle': re.compile(r'.*_Left_middle_finger\.BMP$'),
        'ring': re.compile(r'.*_Left_ring_finger\.BMP$'),
        'pinkie': re.compile(r'.*_Left_little_finger\.BMP$'),
    }
    right_patterns = {
        'thumb': re.compile(r'.*_Right_thumb_finger\.BMP$'),
        'index': re.compile(r'.*_Right_index_finger\.BMP$'),
        'middle': re.compile(r'.*_Right_middle_finger\.BMP$'),
        'ring': re.compile(r'.*_Right_ring_finger\.BMP$'),
        'pinkie': re.compile(r'.*_Right_little_finger\.BMP$'),
    }

    img_folder = os.path.join(abs_path, "images")

    # Create main gender directories in project directory
    if not os.path.exists(male_dirname):
        os.mkdir(male_dirname)
    if not os.path.exists(female_dirname):
        os.mkdir(female_dirname)

    # Create hand and finger subdirectories
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

    global re_exceptions
    # Process each image
    for img in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img)
        
        if os.path.isfile(img_path):
            # Determine gender based on the filename
            if male_pattern.match(img):
                hand = 'Left' if 'Left' in img else 'Right'
                finger_found = False
                for finger, pattern in (left_patterns if hand == 'Left' else right_patterns).items():
                    if pattern.match(img):
                        dest_path = os.path.join(male_dirname, f"{hand} Hand", finger)
                        shutil.copy(img_path, os.path.join(dest_path, img))
                        finger_found = True
                        break
                if not finger_found:
                    shutil.copy(img_path, male_dirname)
            elif female_pattern.match(img):
                hand = 'Left' if 'Left' in img else 'Right'
                finger_found = False
                for finger, pattern in (left_patterns if hand == 'Left' else right_patterns).items():
                    if pattern.match(img):
                        dest_path = os.path.join(female_dirname, f"{hand} Hand", finger)
                        shutil.copy(img_path, os.path.join(dest_path, img))
                        finger_found = True
                        break
                if not finger_found:
                    shutil.copy(img_path, female_dirname)
            else:
                re_exceptions += 1
    print("Number of images that couldn't be categorized: ", re_exceptions)

def processImages(X_train, Y_train, X_test, Y_test):
    # get images (start with female first just to test)                                                 done
    # perform transformations on image                                                                  half-done
    # put image inside of the X_train array or some other intermediate value                            half-done
    # zip X_train and Y_train or some other intermediate array
    # perform randomization on the previously mentioned array
    # put 80% of the images into the train section and 20% into the validation section

    temp_X_array = []
    temp_Y_array = [] 
    f_img_folder = os.path.join(abs_path,  female_dirname)
    for hand_folder in os.listdir(f_img_folder):
        hand_folder_path = os.path.join(f_img_folder, hand_folder)
        for finger_folder in os.listdir(hand_folder_path):
            finger_folder_path = os.path.join(hand_folder_path, finger_folder)
            for img_file in os.listdir(finger_folder_path): 
                img = os.path.join(finger_folder_path, img_file)
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print('image size: ', img.shape)
                temp_X_array.append([img])
                temp_Y_array.append(finger_folder)
            # print('length of temp array: ', len(temp_X_array))
            # temp_Y_array.append(finger_folder)
            #break
    print('length of temp X array: ', len(temp_X_array))
    print('length of temp Y array: ', len(temp_Y_array))
    

def main():
    X_train = np.array([])
    Y_train = np.array([])
    X_test = np.array([])
    Y_test = np.array([])
    # createDirectories() # only need to call once
    processImages(X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()

