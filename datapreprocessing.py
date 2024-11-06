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
import pickle
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
        gender = "Male" if "Male Fingers" in gender_folder_name else "Female"
        for hand_folder in os.listdir(folder_name):
            hand_folder_path = os.path.join(folder_name, hand_folder)
            for finger_folder in os.listdir(hand_folder_path):
                finger_folder_path = os.path.join(hand_folder_path, finger_folder)
                for img_file in os.listdir(finger_folder_path): 
                    img_path = os.path.join(finger_folder_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (96, 103))
                    temp_X_array.append(img)  # Append the image array directly
                    encoded_label = one_hot_encode(hand_folder, finger_folder, gender)
                    temp_Y_array.append(encoded_label)  # Label for the image
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

def one_hot_encode(hand, finger, gender):
    # Initialize a 12-element vector with zeros
    encoding = [0] * 12
    
    # Mapping of fingers for each hand to vector index positions
    finger_mapping = {
        'Left_thumb': 0,
        'Left_index': 1,
        'Left_middle': 2,
        'Left_ring': 3,
        'Left_pinkie': 4,
        'Right_thumb': 9,
        'Right_index': 8,
        'Right_middle': 7,
        'Right_ring': 6,
        'Right_pinkie': 5,
    }
    
    # Set the appropriate finger-hand element to 1
    finger_hand_key = f"{hand}_{finger}"
    if finger_hand_key in finger_mapping:
        encoding[finger_mapping[finger_hand_key]] = 1

    # Set the gender element to 1 (element 10 for male, 11 for female)
    if gender == "Male":
        encoding[10] = 1
    elif gender == "Female":
        encoding[11] = 1

    return encoding

def randomizeImages(temp_X_array, temp_Y_array, X_train, Y_train, X_test, Y_test):
    zipped_array = list(zip(temp_X_array, temp_Y_array))  # Correctly zip the arrays together
    random.shuffle(zipped_array)
    
    # split data into test and train
    test_train_split = int(0.8 * len(zipped_array))
    train_data = zipped_array[:test_train_split]
    test_data = zipped_array[test_train_split:]
    
    # Separate images and labels for training and testing
    X_train, Y_train = zip(*train_data)
    X_test, Y_test = zip(*test_data)
    
    return list(X_train), list(Y_train), list(X_test), list(Y_test)

def main():
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    
    temp_X_array = []
    temp_Y_array = []
    # createDirectories() # only need to call once
    processImages(temp_X_array, temp_Y_array)
    print('Length of temp X array: ', len(temp_X_array))
    print('Length of temp Y array: ', len(temp_Y_array))
    X_train, Y_train, X_test, Y_test = randomizeImages(temp_X_array, temp_Y_array, X_train, Y_train, X_test, Y_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)    

    # NEED TO TEST THIS AND ADD PICKE CODE THEN MOVE ON TO MODEL
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    
    # Put split data into respective pickle
    pickle_out = open("X_train.pickle", "wb")
    pickle.dump(X_train, pickle_out)
    pickle_out.close()

    pickle_out = open("Y_train.pickle", "wb")
    pickle.dump(Y_train, pickle_out)
    pickle_out.close()

    pickle_out = open("X_test.pickle", "wb")
    pickle.dump(X_test, pickle_out)
    pickle_out.close()

    pickle_out = open("Y_test.pickle", "wb")
    pickle.dump(Y_test, pickle_out)
    pickle_out.close()
    
if __name__ == "__main__":
    main()

