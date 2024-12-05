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

if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)} is available")
else:
    print('\nNo GPU available')

device = torch.device("cuda")

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
X_test = torch.tensor(X_test).float().to(device)
Y_test = torch.tensor(Y_test).long().to(device)
X_train = X_train.unsqueeze(1).to(device)
X_test = X_test.unsqueeze(1).to(device)
print('X_train size = ', X_train.size())

Y_subject = Y_train[:, 0].long().to(device)
Y_finger = Y_train[:, 1:6].float().to(device)
Y_hand = Y_train[:, 6].long().to(device)

# Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=2, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)        # Fully connected layers
        # Separate output layers
        self.subject_output = nn.Linear(32, 500)  # 500 subjects
        self.finger_output = nn.Linear(32, 5)    # 5 fingers
        self.hand_output = nn.Linear(32, 2)     # 2 hands (left, right)
    def forward(self, x):
        # Pass through convolutional layers
        x = self.maxpool(torch.relu(self.conv1(x)))
        x = self.maxpool(torch.relu(self.conv2(x)))
        x = self.maxpool(torch.relu(self.conv3(x)))
        
        # Flatten the output
        x = x.view(x.size(0), -1)  # Flatten to shape (batch_size, num_features)
        # Pass through fully connected layers
        # x = torch.relu(self.fc1(x))
        # Separate outputs
        subject_out = (self.subject_output(x))
        finger_out = self.finger_output(x)
        hand_out = self.hand_output(x)
        
        return subject_out, finger_out, hand_out
        
# Accuracy Calculation
def calculate_accuracy(y_pred, y_true):
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
model = NeuralNet().to(device)
loss_fn_subject = nn.CrossEntropyLoss()
loss_fn_finger = nn.CrossEntropyLoss()
loss_fn_hand = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

outputs = model(X_train)

#num_features = outputs.shape[1]

#print(num_features)


# Training Loop
batch_size = 1  # Experiment with this size; adjust based on available GPU memory
num_batches = (X_train.size(0) + batch_size - 1) // batch_size  # Calculate total number of batches

# Training Loop
num_epochs = 500
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    subject_acc_sum = 0
    finger_acc_sum = 0
    hand_acc_sum = 0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, X_train.size(0))

        X_batch = X_train[start_idx:end_idx]
        Y_subject_batch = Y_subject[start_idx:end_idx]
        Y_finger_batch = Y_finger[start_idx:end_idx]
        Y_hand_batch = Y_hand[start_idx:end_idx]

        optimizer.zero_grad()

        # Forward pass
        subject_out, finger_out, hand_out = model(X_batch)
        
        # Compute losses
        loss_subject = loss_fn_subject(subject_out, Y_subject_batch)
        loss_finger = loss_fn_finger(finger_out, Y_finger_batch)
        loss_hand = loss_fn_hand(hand_out, Y_hand_batch)
        
        # Total loss
        loss = loss_subject + loss_finger + loss_hand
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accuracy calculations
        subject_acc_sum += calculate_accuracy(subject_out, Y_subject_batch)
        finger_acc_sum += finger_calculate_accuracy(finger_out, Y_finger_batch)
        hand_acc_sum += calculate_accuracy(hand_out, Y_hand_batch)

    # Average metrics over batches
    avg_loss = total_loss / num_batches
    avg_subject_acc = subject_acc_sum / num_batches
    avg_finger_acc = finger_acc_sum / num_batches
    avg_hand_acc = hand_acc_sum / num_batches

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Loss: {avg_loss:.4f}, '
          f'Subject Accuracy: {avg_subject_acc * 100:.2f}%, '
          f'Finger Accuracy: {avg_finger_acc * 100:.2f}%, '
          f'Hand Accuracy: {avg_hand_acc * 100:.2f}%')

    # Summary writer
    writer.add_scalar('Loss/Total', avg_loss, epoch)
    writer.add_scalar('Loss/Subject', loss_subject.item(), epoch)
    writer.add_scalar('Loss/Finger', loss_finger.item(), epoch)
    writer.add_scalar('Loss/Hand', loss_hand.item(), epoch)

    writer.add_scalar('Accuracy/Subject', avg_subject_acc, epoch)
    writer.add_scalar('Accuracy/Finger', avg_finger_acc, epoch)
    writer.add_scalar('Accuracy/Hand', avg_hand_acc, epoch)

writer.close()

Y_test_subject = Y_test[:, 0].long().to(device)
Y_test_finger = Y_test[:, 1:6].float().to(device)
Y_test_hand = Y_test[:, 6].long().to(device)

with torch.no_grad():
    subject_out, finger_out, hand_out = model(X_test)

    subject_acc = calculate_accuracy(subject_out, Y_test_subject)
    finger_acc = finger_calculate_accuracy(finger_out, Y_test_finger)
    hand_acc = calculate_accuracy(hand_out, Y_test_hand)

    test_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (f"{test_date_time} - "
                 f"Subject Accuracy: {subject_acc * 100:.2f}%, "
                 f"Finger Accuracy: {finger_acc * 100:.2f}%, "
                 f"Hand Accuracy: {hand_acc * 100:.2f}%\n")

    # Append to test_accuracies.txt
    with open("test_accuracies.txt", "a") as f:
        f.write(log_entry)

    print(f"Test Results logged: {log_entry}")