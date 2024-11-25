import torch
import torch.nn as nn
import torch.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime

os.chdir("CSCI158Project")
writer = SummaryWriter()

with open("X_train.pickle", "rb") as f:
    X_train = pickle.load(f)

with open("Y_train.pickle", "rb") as f:
    Y_train = pickle.load(f)

with open("X_test.pickle", "rb") as f:
    X_test = pickle.load(f)

with open("Y_test.pickle", "rb") as f:
    Y_test = pickle.load(f)

Y_train = np.array(Y_train)
X_train = np.array(X_train)
#print("\n\n Y_train np array = ", Y_train.dtype)
#print("\n\n X_train type = ", X_train.dtype)
X_train = torch.tensor(X_train).float()
Y_train = torch.tensor(Y_train).long()
print("\n\n Y_train type in tensor = ", Y_train.dtype)
X_test = torch.tensor(X_test).float()
Y_test = torch.tensor(Y_test).long()
X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)


Y_subject = Y_train[:, 0].long()
Y_finger = Y_train[:, 1:6].float()
Y_hand = Y_train[:, 6].long()

print('\n\nt1: ', Y_subject)
print('\n\nt2: ', Y_finger)
print('\n\nt3: ', Y_hand)

# Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(6720, 512)  # Adjust input size based on flattened output of conv layers
        self.fc2 = nn.Linear(512, 256)

        # Separate output layers
        self.subject_output = nn.Linear(256, 500)  # 500 subjects
        self.finger_output = nn.Linear(256, 5)    # 5 fingers
        self.hand_output = nn.Linear(256, 2)     # 2 hands (left, right)

    def forward(self, x):
        # Pass through convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Flatten the output
        x = x.view(x.size(0), -1)  # Flatten to shape (batch_size, num_features)
        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Separate outputs
        subject_out = self.subject_output(x)
        finger_out = self.finger_output(x)
        hand_out = self.hand_output(x)
        
        return subject_out, finger_out, hand_out
        
# Accuracy Calculation
def calculate_accuracy(y_pred, y_true):
    """Calculate accuracy for classification tasks"""
    _, predicted = torch.max(y_pred, 1)  # Get predicted class indices
    correct = (predicted == y_true).sum().item()
    return correct / y_true.size(0)

# Finger accuracy calculation
def finger_calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    _, true = torch.max(y_true, 1)
    correct = (predicted == true).sum().item()
    return correct / true.size(0)


# Dummy Data Preparation
# Assume we have preprocessed dataset with features (X) and labels split into subject, finger, hand
#X_train = torch.rand(16000, 1, 103, 96)  # Example input: batch of 32 images, 1 channel, 28x28 resolution
#Y_subject = torch.randint(0, 500, (16000,))  # Subject labels (500 classes)
#Y_finger = torch.randint(0, 5, (16000,))     # Finger labels (5 classes)
#Y_hand = torch.randint(0, 2, (16000,))       # Hand labels (binary)

# Model, Loss, and Optimizer
model = NeuralNet()
loss_fn_subject = nn.CrossEntropyLoss()
loss_fn_finger = nn.CrossEntropyLoss()
loss_fn_hand = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

outputs = model(X_train)

#num_features = outputs.shape[1]

#print(num_features)


# Training Loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    subject_out, finger_out, hand_out = model(X_train)
    # Compute losses
    loss_subject = loss_fn_subject(subject_out, Y_subject)
    loss_finger = loss_fn_finger(finger_out, Y_finger)
    loss_hand = loss_fn_hand(hand_out, Y_hand)
    # Total loss
    loss = loss_subject + loss_finger + loss_hand

    # Summary writer
    writer.add_scalar('Loss/Total', loss.item(), epoch)
    writer.add_scalar('Loss/Subject', loss_subject.item(), epoch)
    writer.add_scalar('Loss/Finger', loss_finger.item(), epoch)
    writer.add_scalar('Loss/Hand', loss_hand.item(), epoch)
    
    writer.add_scalar('Accuracy/Subject', subject_acc, epoch)
    writer.add_scalar('Accuracy/Finger', finger_acc, epoch)
    writer.add_scalar('Accuracy/Hand', hand_acc, epoch)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Accuracy calculations
    subject_acc = calculate_accuracy(subject_out, Y_subject)
    finger_acc = finger_calculate_accuracy(finger_out, Y_finger)
    hand_acc = calculate_accuracy(hand_out, Y_hand)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Loss: {loss.item():.4f}, '
          f'Subject Accuracy: {subject_acc * 100:.2f}%, '
          f'Finger Accuracy: {finger_acc * 100:.2f}%, '
          f'Hand Accuracy: {hand_acc * 100:.2f}%')

writer.close()

Y_test_subject = Y_train[:, 0].long()
Y_test_finger = Y_train[:, 1:6].float()
Y_test_hand = Y_train[:, 6].long()

with torch.no_grad():
    subject_out, finger_out, hand_out = model(X_test)

    calculate_accuracy(subject_out, Y_test_subject)
    finger_calculate_accuracy(finger_out, Y_test_finger)
    calculate_accuracy(hand_out, Y_test_hand)

    test_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (f"{test_date_time} - "
                 f"Subject Accuracy: {subject_acc * 100:.2f}%, "
                 f"Finger Accuracy: {finger_acc * 100:.2f}%, "
                 f"Hand Accuracy: {hand_acc * 100:.2f}%\n")

    # Append to test_accuracies.txt
    with open("test_accuracies.txt", "a") as f:
        f.write(log_entry)

    print(f"Test Results logged: {log_entry}")