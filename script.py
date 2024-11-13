import torch
import torch.nn as nn
import torch.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("CSCI158Project")
print('cur dir: ', os.getcwd())

with open("X_train.pickle", "rb") as f:
    X_train = pickle.load(f)

with open("Y_train.pickle", "rb") as f:
    Y_train = pickle.load(f)

with open("X_test.pickle", "rb") as f:
    X_test = pickle.load(f)

with open("Y_test.pickle", "rb") as f:
    Y_test = pickle.load(f)

X_train = torch.tensor(X_train)
Y_train = torch.tensor(Y_train)
X_test = torch.tensor(X_test)
Y_test = torch.tensor(Y_test)
X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=2, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 106 * 99, 512)
        self.fc2 = nn.Linear(512, 256)  
        self.fc3 = nn.Linear(256, 128)  
        self.fc4 = nn.Linear(128, 10) # output layer
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x) # output layer
        return x
    
model = NeuralNet()

dummy = torch.randn(1, 1, 103, 96)

output = model(dummy)

num_features = output.shape[1]

print("number of output features = ", num_features)

# Create training loop, output accuracy and loss, choose loss function and optimizer

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

print(len(X_train), "\n")
print(len(Y_train), "\n")
print(len(X_test), "\n")
print(len(Y_test), "\n")

