# helper_lib/assignment2.py
 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Assignment2_CNN(nn.Module):
    def __init__(self):
        super(Assignment2_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # Apply first conv layer and pooling
        x = self.pool(F.relu(self.conv2(x)))   # Apply second conv layer and pooling
        x = x.view(-1, 32 * 16 * 16)           # Flatten the tensor
        x = F.relu(self.fc1(x))                # First fully connected layer
        x = self.fc2(x)                        # Second fully connected layer
        return x
    

