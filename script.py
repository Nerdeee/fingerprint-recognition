import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
import datetime

writer = SummaryWriter()

# Load data
with open("X_train.pickle", "rb") as f:
    X_train = pickle.load(f)

with open("Y_train.pickle", "rb") as f:
    Y_train = pickle.load(f)

with open("X_test.pickle", "rb") as f:
    X_test = pickle.load(f)

with open("Y_test.pickle", "rb") as f:
    Y_test = pickle.load(f)

# Convert to torch tensors
X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
Y_train = torch.tensor(np.array(Y_train), dtype=torch.long)

X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
Y_test = torch.tensor(np.array(Y_test), dtype=torch.long)

# Unsqueeze to add a channel dimension (assuming grayscale input)
X_train = X_train.unsqueeze(1)  # Shape: (N, 1, H, W)
X_test = X_test.unsqueeze(1)    # Shape: (N, 1, H, W)

# Separate targets (subject, finger, hand)
Y_subject = Y_train[:, 0].long()
Y_finger = Y_train[:, 1:6].float()  # One-hot encoded or regression
Y_hand = Y_train[:, 6].long()

# Dataset and DataLoader
train_dataset = TensorDataset(X_train, Y_subject, Y_finger, Y_hand)
test_dataset = TensorDataset(X_test, Y_test[:, 0].long(), Y_test[:, 1:6].float(), Y_test[:, 6].long())

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Batch size 32 is typical
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc2 = nn.Linear(4608, 1024)

        # Separate output layers
        self.subject_output = nn.Linear(1024, 500)  # 500 subjects
        self.finger_output = nn.Linear(1024, 5)     # 5 fingers
        self.hand_output = nn.Linear(1024, 2)       # 2 hands (left, right)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv3(x))
        x = self.maxpool(x)

        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)  # Flatten to shape (batch_size, num_features)
        
        x = torch.relu(self.fc2(x))

        # Separate outputs for subject, finger, and hand
        subject_out = self.subject_output(x)
        finger_out = self.finger_output(x)
        hand_out = self.hand_output(x)
        
        return subject_out, finger_out, hand_out

# Accuracy Calculation
def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    return correct / y_true.size(0)

def finger_calculate_accuracy(y_pred, y_true):
    # Get the predicted class indices
    _, predicted = torch.max(y_pred, 1)
    # If the ground truth is one-hot encoded (y_true has shape [batch_size, 5]), 
    # we need to extract the indices where the value is 1.
    # In this case, we use torch.argmax to get the indices of the max value (one-hot labels).
    true = torch.argmax(y_true, 1)  # Converts one-hot labels to class indices
    correct = (predicted == true).sum().item()  # Count how many are correct
    return correct / y_true.size(0)

# Training Loop
model = NeuralNet()
loss_fn_subject = nn.CrossEntropyLoss()
loss_fn_finger = nn.CrossEntropyLoss()  # If finger is multi-class
loss_fn_hand = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_subject_acc = 0
    total_finger_acc = 0
    total_hand_acc = 0

    for batch_idx, (inputs, subjects, fingers, hands) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        subject_out, finger_out, hand_out = model(inputs)

        # Compute losses
        loss_subject = loss_fn_subject(subject_out, subjects)
        loss_finger = loss_fn_finger(finger_out, fingers)
        loss_hand = loss_fn_hand(hand_out, hands)
        loss = loss_subject + loss_finger + loss_hand

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate accuracy for the batch
        subject_acc = calculate_accuracy(subject_out, subjects)
        finger_acc = finger_calculate_accuracy(finger_out, fingers)
        hand_acc = calculate_accuracy(hand_out, hands)

        # Update statistics
        total_loss += loss.item()
        total_subject_acc += subject_acc
        total_finger_acc += finger_acc
        total_hand_acc += hand_acc

    avg_loss = total_loss / len(train_loader)
    avg_subject_acc = total_subject_acc / len(train_loader)
    avg_finger_acc = total_finger_acc / len(train_loader)
    avg_hand_acc = total_hand_acc / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Loss: {avg_loss:.4f}, "
          f"Subject Accuracy: {avg_subject_acc*100:.2f}%, "
          f"Finger Accuracy: {avg_finger_acc*100:.2f}%, "
          f"Hand Accuracy: {avg_hand_acc*100:.2f}%")

    # Tensorboard logging
    writer.add_scalar('Loss/Total', avg_loss, epoch)
    writer.add_scalar('Accuracy/Subject', avg_subject_acc, epoch)
    writer.add_scalar('Accuracy/Finger', avg_finger_acc, epoch)
    writer.add_scalar('Accuracy/Hand', avg_hand_acc, epoch)

writer.close()

# Evaluate on the test set
model.eval()
with torch.no_grad():
    total_subject_acc = 0
    total_finger_acc = 0
    total_hand_acc = 0

    for inputs, subjects, fingers, hands in test_loader:
        subject_out, finger_out, hand_out = model(inputs)

        subject_acc = calculate_accuracy(subject_out, subjects)
        finger_acc = finger_calculate_accuracy(finger_out, fingers)
        hand_acc = calculate_accuracy(hand_out, hands)

        total_subject_acc += subject_acc
        total_finger_acc += finger_acc
        total_hand_acc += hand_acc

    avg_subject_acc = total_subject_acc / len(test_loader)
    avg_finger_acc = total_finger_acc / len(test_loader)
    avg_hand_acc = total_hand_acc / len(test_loader)

    test_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (f"{test_date_time} - "
                 f"Subject Accuracy: {avg_subject_acc*100:.2f}%, "
                 f"Finger Accuracy: {avg_finger_acc*100:.2f}%, "
                 f"Hand Accuracy: {avg_hand_acc*100:.2f}%\n")

    with open("test_accuracies.txt", "a") as f:
        f.write(log_entry)

    print(f"Test Results logged: {log_entry}")
