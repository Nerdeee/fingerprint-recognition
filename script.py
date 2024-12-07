import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
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

# convert to torch tensors
X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
Y_train = torch.tensor(np.array(Y_train), dtype=torch.long)

X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
Y_test = torch.tensor(np.array(Y_test), dtype=torch.long)

# unsqueezes to add a channel dimension for greyscale images
X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)

# separates targets (subject, finger, hand)
Y_subject = Y_train[:, 0].long()
Y_finger = Y_train[:, 1:6].float()
Y_hand = Y_train[:, 6].long()

# loads dataset and creates dataloader
train_dataset = TensorDataset(X_train, Y_subject, Y_finger, Y_hand)
test_dataset = TensorDataset(X_test, Y_test[:, 0].long(), Y_test[:, 1:6].float(), Y_test[:, 6].long())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.fc2 = nn.Linear(294912, 1024)

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

        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc2(x))

        subject_out = self.subject_output(x)
        finger_out = self.finger_output(x)
        hand_out = self.hand_output(x)
        
        return subject_out, finger_out, hand_out

# calculates accuracy for subject and hand
def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    return correct / y_true.size(0)

# calculates accuracy for finger
def finger_calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    true = torch.argmax(y_true, 1)
    correct = (predicted == true).sum().item()
    return correct / y_true.size(0)

# Dummy Data Preparation
#X_train = torch.rand(16000, 1, 103, 96)
#Y_subject = torch.randint(0, 500, (16000,))
#Y_finger = torch.randint(0, 5, (16000,))
#Y_hand = torch.randint(0, 2, (16000,))

model = NeuralNet().to(device)
loss_fn_subject = nn.CrossEntropyLoss() # loss functions
loss_fn_finger = nn.CrossEntropyLoss()
loss_fn_hand = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    total_subject_acc = 0
    total_finger_acc = 0
    total_hand_acc = 0

    for batch_idx, (inputs, subjects, fingers, hands) in enumerate(train_loader):
        inputs = inputs.to(device)
        subjects = subjects.to(device)
        fingers = fingers.to(device)
        hands = hands.to(device)
        
        optimizer.zero_grad()

        subject_out, finger_out, hand_out = model(inputs)

        loss_subject = loss_fn_subject(subject_out, subjects)   # calculate loss for each metric
        loss_finger = loss_fn_finger(finger_out, fingers)
        loss_hand = loss_fn_hand(hand_out, hands)
        loss = loss_subject + loss_finger + loss_hand

        loss.backward()
        optimizer.step()

        # calculate accuracy for each metric
        subject_acc = calculate_accuracy(subject_out, subjects) 
        finger_acc = finger_calculate_accuracy(finger_out, fingers)
        hand_acc = calculate_accuracy(hand_out, hands)

        total_loss += loss.item()
        total_subject_acc += subject_acc
        total_finger_acc += finger_acc
        total_hand_acc += hand_acc

    # get average loss across all of the batches in the epoch
    avg_loss = total_loss / len(train_loader)
    # average accuracy across all of the batches in the epoch
    avg_subject_acc = total_subject_acc / len(train_loader)
    avg_finger_acc = total_finger_acc / len(train_loader)
    avg_hand_acc = total_hand_acc / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Loss: {avg_loss:.4f}, "
          f"Subject Accuracy: {avg_subject_acc*100:.2f}%, "
          f"Finger Accuracy: {avg_finger_acc*100:.2f}%, "
          f"Hand Accuracy: {avg_hand_acc*100:.2f}%")

    # tensorboard logging
    writer.add_scalar('Total Loss / Epochs', avg_loss, epoch)
    writer.add_scalar('Subject Accuracy / Epochs', avg_subject_acc, epoch)
    writer.add_scalar('Finger Accuracy / Epochs', avg_finger_acc, epoch)
    writer.add_scalar('Hand Accuracy / Epochs', avg_hand_acc, epoch)

writer.close()

# test loop
model.eval()
with torch.no_grad():
    total_subject_acc = 0
    total_finger_acc = 0
    total_hand_acc = 0

    for inputs, subjects, fingers, hands in test_loader:
        inputs = inputs.to(device)
        subjects = subjects.to(device)
        fingers = fingers.to(device)
        hands = hands.to(device)
        
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
