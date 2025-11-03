# helper_lib/energy_model.py

import torch
import torch.nn as nn

def swish(x):
    return x * torch.sigmoid(x)

class EnergyModel(nn.Module):
    def __init__(self, in_channels=3):     # RGB default for CIFAR-10
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


# ---- Langevin sampling (fixed for RGB 3-channel images) ----
def langevin_sample(model, num_samples=4, steps=1000, step_size=0.02, noise_scale=0.005, device="cpu"):
    """
    Langevin dynamics sampling for EBM.
    """
    model.eval()
    x = torch.randn(num_samples, 3, 32, 32, device=device).clamp(-1, 1)  # 3 channels for CIFAR
    x.requires_grad_(True)

    for _ in range(steps):
        e = model(x).sum()
        grad = torch.autograd.grad(e, x)[0]
        x = (x - step_size * grad + noise_scale * torch.randn_like(x)).clamp(-1, 1)
        x = x.detach().requires_grad_(True)

    return x.detach()






