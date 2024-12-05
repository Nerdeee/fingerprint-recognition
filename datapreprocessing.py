import numpy as np
import cv2
import os
import random
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

def get_finger_info(filename):
    """Extract subject, hand, finger, and orientation from filename."""
    parts = filename.split('_')
    subject = int(parts[0])  # e.g., '008'
    hand_finger = parts[1]  # R1, L2, etc.
    hand = hand_finger[0]  # R or L
    orientation = hand_finger[1]
    finger = int(parts[2].split('.')[0])
    return subject, hand, finger, orientation

def label_encode(subject, hand, finger):
    """
    Create one-hot encoding with 9 elements:
    - Indices 0-2: subject number (three digits)
    - Indices 3-7: finger (3=thumb, 4=index, 5=middle, 6=ring, 7=pinkie)
    - Index 8: hand (0=left, 1=right)
    """
    encoding = [0] * 7
    
    encoding[0] = subject  # Encode subject (3 digits)
    
    encoding[finger + 1] = 1  # Encode finger (1-5 maps to indices 3-7)
    
    encoding[6] = 1 if hand == 'R' else 0  # Encode hand (0 for left, 1 for right)
    
    return encoding

def transformImage(img):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    
    # Normalize the image
    normalized_image = image / 255.0
    
    # Resize the image to the desired size
    resized_image = cv2.resize(normalized_image, (96, 96))
    
    # Step 1: Histogram Equalization
    equalized = cv2.equalizeHist((resized_image * 255).astype(np.uint8))
    
    # Step 2: Gaussian Blur (Optional for noise reduction)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # Step 3: Adaptive Thresholding
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    
    # Step 4: Morphological Opening (Remove small noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    
    # Step 5: Skeletonization
    def skeletonize(image):
        skeleton = np.zeros(image.shape, dtype=np.uint8)
        temp = np.copy(image)
        while True:
            eroded = cv2.erode(temp, None)
            dilated = cv2.dilate(eroded, None)
            subtracted = cv2.subtract(temp, dilated)
            skeleton = cv2.bitwise_or(skeleton, subtracted)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        return skeleton
    
    skeletonized = skeletonize(opened)

    return skeletonized

def process_dataset(root_dir):
    """Process the dataset and split by subject and finger."""
    subject_data = defaultdict(lambda: defaultdict(list))  # Store images by subject and finger
    
    # Process all subjects
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
    
    # Now we perform the splitting into training and testing
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    
    for subject, fingers in subject_data.items():
        for finger, data in fingers.items():
            # Shuffle the data for this subject's finger
            random.shuffle(data)
            
            # One photo for testing, three for training
            test_img, train_data = data[0], data[1:]
            
            # Add test image to test set
            X_test.append(test_img[0])
            Y_test.append(test_img[1])
            
            # Add remaining training images to train set
            for img, label, hand in train_data:
                X_train.append(img)
                Y_train.append(label)
    
    # Shuffle the training data
    zipped_train = list(zip(X_train, Y_train))
    random.shuffle(zipped_train)
    X_train, Y_train = zip(*zipped_train)
    X_train, Y_train = list(X_train), list(Y_train)
    
    # Convert to numpy arrays
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

    # Process the dataset
    X_train, Y_train, X_test, Y_test = process_dataset(root_dir)
    print('\nMain function Y_train = ', Y_train)
    
    # Save to pickle files
    save_to_pickle(X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()
