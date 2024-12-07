import numpy as np
import cv2
import os
import random
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def get_finger_info(filename):
    parts = filename.split('_')
    subject = int(parts[0])
    hand_finger = parts[1]
    hand = hand_finger[0]
    orientation = hand_finger[1]
    finger = int(parts[2].split('.')[0])
    return subject, hand, finger, orientation

def label_encode(subject, hand, finger):
    """
    Encoding with 9 elements:
    - Indices 0-2: subject number (three digits)
    - Indices 3-7: finger (3=thumb, 4=index, 5=middle, 6=ring, 7=pinkie)
    - Index 8: hand (0=left, 1=right)
    """
    encoding = [0] * 7
    
    encoding[0] = subject
    
    encoding[finger + 1] = 1
    
    encoding[6] = 1 if hand == 'R' else 0
    
    return encoding

def transformImage(img):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # images resized to 96 x 96 pixels
    resized_image = cv2.resize(image, (96, 96))
    # histogram equalization redistributed the intensity of the pixel values to improve contrast
    equalized = cv2.equalizeHist((resized_image).astype(np.uint8))
    # gaussian blur used for noise reduction
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    # adaptive thresholding for converting the grayscale image into a binary one
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 7, 2)
    # morphological filtering to remove noise and increase the contours of the fingerprints
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    # skeletonization to reduce the fingerprints to 1 pixel wide structures
    skeletonized = skeletonize(opened)

    return skeletonized

# processes the dataset and splits by subject and finger
def process_dataset(root_dir):
    subject_data = defaultdict(lambda: defaultdict(list))
    
    images_dir = os.path.join(root_dir, "images")
    for subject_folder in os.listdir(images_dir):
        if not subject_folder.isdigit():
            continue
            
        subject_path = os.path.join(images_dir, subject_folder)
        
        for hand in ['L', 'R']:
            hand_path = os.path.join(subject_path, hand)
            if not os.path.exists(hand_path):
                continue
                
            for img_name in os.listdir(hand_path):
                if not img_name.lower().endswith('.bmp'):
                    continue
                    
                subject, hand, finger, orientation = get_finger_info(img_name)
                
                img_path = os.path.join(hand_path, img_name)
                img = transformImage(img_path)
                
                label = label_encode(subject, hand, finger)
                
                subject_data[subject][finger].append((img, label, hand))
                print(f'{subject} {hand} {finger} {orientation} processed successfully')
    
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    
    # selects one finger variation at random for testing and the rest of the images for training for each finger
    for subject, fingers in subject_data.items():
        for finger, data in fingers.items():
            random.shuffle(data)
            test_img, train_data = data[0], data[1:]
            X_test.append(test_img[0])
            Y_test.append(test_img[1])
            for img, label, hand in train_data:
                X_train.append(img)
                Y_train.append(label)
    
    # shuffles the training data
    zipped_train = list(zip(X_train, Y_train))
    random.shuffle(zipped_train)
    X_train, Y_train = zip(*zipped_train)
    X_train, Y_train = list(X_train), list(Y_train)
    
    return np.array(X_train, dtype=int), np.array(Y_train, dtype=int), np.array(X_test, dtype=int), np.array(Y_test, dtype=int)

def save_to_pickle(X_train, Y_train, X_test, Y_test):
    """Save processed data to pickle files."""
    datasets = {
        "X_train.pickle": X_train,
        "Y_train.pickle": Y_train,
        "X_test.pickle": X_test,
        "Y_test.pickle": Y_test
    }
    for filename, data in datasets.items():
        with open(filename, "wb") as f:
            pickle.dump(data, f)

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    X_train, Y_train, X_test, Y_test = process_dataset(root_dir)
    print('\nMain function Y_train = ', Y_train)
    save_to_pickle(X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()
