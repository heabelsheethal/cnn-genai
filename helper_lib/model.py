import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.models import resnet18 

NUM_CLASSES = 10

# ================================
# SIMPLE FCNN MODEL
# ================================
class MLP(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(MLP, self).__init__()     
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x


# ================================
# SIMPLE CNN MODEL
# ================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ================================
# ENHANCED CNN MODEL
# ================================
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



def get_model(model_name="FCNN"):
    if model_name == "FCNN":
        return MLP()
    elif model_name == "CNN":
        return SimpleCNN()
    elif model_name == "EnhancedCNN":
        return EnhancedCNN()
    else:
        raise ValueError(f"Unknown model: {model_name}")





# ================================
# MODEL OUTPUTS
# ================================

# -------FCNN MODEL-------
# Finished epoch 5: loss=2.0659, accuracy=0.388
# Test Accuracy: 0.372
# Average Loss: 2.0827

# -------CNN MODEL-------
# Finished epoch 5: loss=0.9409, accuracy=0.670
# Test Accuracy: 0.664
# Average Loss: 0.9710

# -------ENHANCED CNN MODEL-------
# Finished epoch 5: loss=0.7658, accuracy=0.735
# Test Accuracy: 0.663
# Average Loss: 1.0314
