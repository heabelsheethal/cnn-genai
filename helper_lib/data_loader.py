import torch
from torchvision import datasets , transforms
from torch.utils.data import DataLoader

def get_data_loader(data_dir, batch_size=32, train=True):
    # TODO: create the data loader
    """
    Loads CIFAR-10 data with basic transforms.
    """

    # Define a transform: convert images into PyTorch tensors (CIFAR-10 images are originally PIL images)
    transform = transforms.Compose([transforms.ToTensor()])

    # Load train or test dataset depending on the flag
    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)

    # Create train or test DataLoader for batching
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    
    return loader


