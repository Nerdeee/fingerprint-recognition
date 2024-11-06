import torch
import torch.nn as nn
import torch.functional as F
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

with open("X_train.pickle", "rb") as f:
    X_train = pickle.load(f)

with open("Y_train.pickle", "rb") as f:
    Y_train = pickle.load(f)

with open("X_test.pickle", "rb") as f:
    X_test = pickle.load(f)

with open("Y_test.pickle", "rb") as f:
    Y_test = pickle.load(f)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
    
    def forward(self, x):
        return

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

print(len(X_train), "\n")
print(len(Y_train), "\n")
print(len(X_test), "\n")
print(len(Y_test), "\n")

