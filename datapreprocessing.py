# I guess we do option 1 since I'm already implementing the data preprocessing lol
# Need to ask if using scikit-image built in filters such as sobel and others are considered built in libraries or if we'll have to implement them ourselves
# One hot encode the labels into one vector. The first 10 elements will encode the left hand from thumb to pinkie, then right hand from pinkie to to thumb, and
# then the last two elements will encode male and female. Essentially there should always be 2 elements that are hot inside the array. 

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import pickle
import re
import shutil
from skimage import io, filters, color

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

#def processImages(X_train, Y_train, X_test, Y_test):
def processImages(temp_X_array, temp_Y_array):

    def traverse(gender_folder_name):
        folder_name = os.path.join(abs_path, gender_folder_name)
        for hand_folder in os.listdir(folder_name):
            hand_folder_path = os.path.join(folder_name, hand_folder)
            for finger_folder in os.listdir(hand_folder_path):
                finger_folder_path = os.path.join(hand_folder_path, finger_folder)
                for img_file in os.listdir(finger_folder_path): 
                    img_path = os.path.join(finger_folder_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (96, 103))
                    temp_X_array.append(img)  # Append the image array directly
                    temp_Y_array.append(finger_folder)  # Label for the image
                    #print('\n\n\n\n\n', temp_Y_array[0], '\n\n\n\n')
                    #break
        print(f'Length of temp X array for {gender_folder_name}: ', len(temp_X_array))
        print(f'Length of temp Y array for {gender_folder_name}: ', len(temp_Y_array))

    def transformImage(img):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        
        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
        
        x = np.zeros_like(img, dtype=float)
        y = np.zeros_like(img, dtype=float)
        img = img.astype(float)  # Ensure the image is in float for calculations
        
        # Apply Sobel filter
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                x[i, j] = np.sum(sobel_x * img[i-1:i+2, j-1:j+2])
                y[i, j] = np.sum(sobel_y * img[i-1:i+2, j-1:j+2])
        
        magnitudes = np.sqrt(x ** 2 + y ** 2)
        
        # Normalize magnitudes
        if np.max(magnitudes) != 0:
            normalized_magnitudes = magnitudes / np.max(magnitudes)
        else:
            normalized_magnitudes = magnitudes  # Avoid division if the max is zero
        
        return normalized_magnitudes

    # Traverse directories for both genders
    traverse(female_dirname)
    traverse(male_dirname)

    # Perform transformations on images in temp_X_array
    for i in range(len(temp_X_array)):
       temp_X_array[i] = transformImage(temp_X_array[i])  # Replace each image with its transformed version
       print(f"index {i}: ", np.array(temp_X_array[i]).shape)
    return

def encodeImages(temp_X_array, temp_Y_array):
    print(type(temp_X_array))
    print(type(temp_Y_array))
    zipped_array = []
    for i in range(len(temp_X_array)):
        zipped_array.append(zip(temp_X_array[i], temp_Y_array[i])) # array of tuples where each tule is of length 2. Tuple[0] = image matrix, tuple[1] = class
    print(len(zipped_array))
    
def main():
    
    X_train = np.array([])
    Y_train = np.array([])
    X_test = np.array([])
    Y_test = np.array([])
    
    temp_X_array = []
    temp_Y_array = []
    # createDirectories() # only need to call once
    # processImages(X_train, Y_train, X_test, Y_test)
    # processImages(zipped_images_labels)
    processImages(temp_X_array, temp_Y_array)
    print('Length of temp X array: ', len(temp_X_array))
    print('Length of temp Y array: ', len(temp_Y_array))
    encodeImages(temp_X_array, temp_Y_array)
    """ print("X_train shape:", X_train)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    """

if __name__ == "__main__":
    main()

