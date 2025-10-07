import torch
from torchvision import datasets , transforms
from torch.utils.data import DataLoader

def get_data_loader(data_dir, batch_size=32, train=True, model_name="CNN_Assignment2"):
    # TODO: create the data loader
    """
    Loads CIFAR-10 data with basic transforms.
    """

    # Define a transform: convert images into PyTorch tensors (CIFAR-10 images are originally PIL images)
    # Determine target image size based on model
    if model_name == "CNN_Assignment2":
        target_size = 64
    else:
        target_size = 32

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),  # Resize images if needed
        transforms.ToTensor()
    ])



    # Load train or test dataset depending on the flag
    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)

    # Create train or test DataLoader for batching
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    
    return loader


