import numpy as np
import cv2
import os
import random
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

def get_finger_info(filename):
    """Extract subject, hand, finger, and orientation from filename."""
    # Example: 008_R1_2.bmp
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
    
    # Encode subject (3 digits)
    #subject_digits = [int(d) for d in subject]
    #for i in range(3):
    #    encoding[i] = subject_digits[i]
    encoding[0] = subject
    # Encode finger (1-5 maps to indices 3-7)
    encoding[finger + 1] = 1  # +2 because finger 1 should map to index 3
    
    # Encode hand (0 for left, 1 for right)
    encoding[6] = 1 if hand == 'R' else 0    
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
    """Process the dataset and split by subject."""
    # Dictionary to store images by subject
    subject_data = defaultdict(list)
    
    # Process all subjects
    images_dir = os.path.join(root_dir, "images")
    for subject_folder in os.listdir(images_dir):
        if not subject_folder.isdigit():
            continue
            
        subject_path = os.path.join(images_dir, subject_folder)
        
        # Process left and right hand folders
        for hand in ['L', 'R']:
            hand_path = os.path.join(subject_path, hand)
            if not os.path.exists(hand_path):
                continue
                
            # Process all finger images in the hand folder
            for img_name in os.listdir(hand_path):
                if not img_name.lower().endswith('.bmp'):
                    continue
                    
                # Get image information
                subject, hand, finger, orientation = get_finger_info(img_name)
                
                # Read and process image
                img_path = os.path.join(hand_path, img_name)
                img = transformImage(img_path)
                # Create label
                label = label_encode(subject, hand, finger) # looks good here
                # Store processed image and label with subject
                subject_data[subject].append((img, label))
                print(f'{subject} {hand} {finger} {orientation} processed successfully')
    
    # Split data into train and test sets by subject
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    print('subject data: ', subject_data[0])
    for subject, data in subject_data.items():
        # Shuffle subject's data
        random.shuffle(data)
        
        # Split 80/20 for this subject
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        # Add to training set
        for img, label in train_data:
            X_train.append(img)
            Y_train.append(label)
            
        # Add to test set
        for img, label in test_data:
            X_test.append(img)
            Y_test.append(label)
    
    zipped_train = list(zip(X_train, Y_train))
    random.shuffle(zipped_train)
    X_train, Y_train = zip(*zipped_train)
    #print('Y_train unpacked = ', Y_train)
    X_train, Y_train = list(X_train), list(Y_train)
    print('Y_train list = ', Y_train)
    # Convert to numpy arrays
    return X_train, Y_train, X_test, Y_test

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
    # Get the root directory (where the script is located)
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Process the dataset
    X_train, Y_train, X_test, Y_test = process_dataset(root_dir)
    print('\nMain function Y_train = ', Y_train)
    X_train, Y_train, X_test, Y_test = np.array(X_train, dtype=int), np.array(Y_train, dtype=int), np.array(X_test, dtype=int), np.array(Y_test, dtype=int)
    print('\nMain function Y_train as numpy array = ', Y_train)
    # Print shapes for verification
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    for i in Y_train:
        print(i, '\n')
    # might want to print the shuffled arrays just to make sure it worked properly

    # Save to pickle files
    save_to_pickle(X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()