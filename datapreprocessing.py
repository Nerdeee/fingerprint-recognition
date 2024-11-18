import numpy as np
import cv2
import os
import random
import pickle
from collections import defaultdict

def get_finger_info(filename):
    """Extract subject, hand, finger, and orientation from filename."""
    # Example: 008_R1_2.bmp
    parts = filename.split('_')
    subject = parts[0]  # e.g., '008'
    hand_finger = parts[1]  # R1, L2, etc.
    hand = hand_finger[0]  # R or L
    finger = int(hand_finger[1])  # 1-5
    orientation = int(parts[2].split('.')[0])
    return subject, hand, finger, orientation

def one_hot_encode(subject, hand, finger):
    """
    Create one-hot encoding with 9 elements:
    - Indices 0-2: subject number (three digits)
    - Indices 3-7: finger (3=thumb, 4=index, 5=middle, 6=ring, 7=pinkie)
    - Index 8: hand (0=left, 1=right)
    """
    encoding = [0] * 9
    
    # Encode subject (3 digits)
    subject_digits = [int(d) for d in subject]
    for i in range(3):
        encoding[i] = subject_digits[i]
    
    # Encode finger (1-5 maps to indices 3-7)
    encoding[finger + 2] = 1  # +2 because finger 1 should map to index 3
    
    # Encode hand (0 for left, 1 for right)
    encoding[8] = 1 if hand == 'R' else 0
        
    return encoding

def transformImage(img):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    normalized_image = image / 255.0
    resized_image = cv2.resize(normalized_image, (96, 103))
    resized_image_uint8 = (resized_image * 255).astype(np.uint8)
    blurred_image = cv2.GaussianBlur(resized_image_uint8, (3, 3), 0)
    _, segmented_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    equalized_image = cv2.equalizeHist(segmented_image)
    binarized_image = cv2.adaptiveThreshold(equalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
    mean, std_dev = np.mean(binarized_image), np.std(binarized_image)
    standardized_image = (binarized_image - mean) / std_dev
    
    return standardized_image

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
                label = one_hot_encode(subject, hand, finger)
                
                # Store processed image and label with subject
                subject_data[subject].append((img, label))
                print(f'{subject} {hand} {finger} {orientation} processed successfully')
    
    # Split data into train and test sets by subject
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    
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
    
    # Convert to numpy arrays
    return (np.array(X_train), np.array(Y_train),
            np.array(X_test), np.array(Y_test))

def save_to_pickle(X_train, Y_train, X_test, Y_test):
    """Save processed data to pickle files."""
    datasets = {
        "X_train.pickle": X_train,
        "Y_train.pickle": Y_train,
        "X_test.pickle": X_test,
        "Y_test.pickle": Y_test
    }
    os.chdir('CSCI158Project')    
    for filename, data in datasets.items():
        with open(filename, "wb") as f:
            pickle.dump(data, f)

def main():
    # Get the root directory (where the script is located)
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Process the dataset
    X_train, Y_train, X_test, Y_test = process_dataset(root_dir)
    
    # Print shapes for verification
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    
    # Save to pickle files
    save_to_pickle(X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()