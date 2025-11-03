# helper_lib/data_loader.py

import torch
import os
from PIL import Image 
from torchvision import datasets , transforms
from torch.utils.data import DataLoader, Dataset

def get_data_loader(data_dir, batch_size=32, train=True, model_name="CNN_Assignment2", target_size = 32):
    # TODO: create the data loader
    """
    Loads and preprocesses datasets for different model types.
        - For CNN / FCNN models: uses CIFAR-10 dataset.
        - For GAN models: uses CelebA dataset.
        - For custom models: uses a custom dataset class.
    """

    # Define a transform: convert images into PyTorch tensors (CIFAR-10 images are originally PIL images)
    # Determine target image size based on model
    if model_name == "CNN_Assignment2":
        target_size = 64

  
    # ===============================================================
    # Case 1: ----------GAN model (from Notes) -> Use CelebA dataset with 64×64 crops-------
    # ===============================================================
    if model_name == "GAN":
        # Define transforms
        transform = transforms.Compose([
                            transforms.Resize(128),                       # Resize images to 128×128
                            transforms.CenterCrop(64),                    # Crop the central 64×64 region
                            transforms.ToTensor(),                        # Convert PIL image → Tensor (0–1)
                            transforms.Normalize([0.5] * 3, [0.5] * 3)       # Normalize to [-1, 1] range for GANs
                        ])
        
        # CelebA dataset supports 'train', 'valid', 'test' splits
        if train:
            split = 'train'
        else:
            split = 'test'
        # Load train or test dataset depending on the flag
        dataset = datasets.CelebA(root=data_dir, split=split, download=True, transform=transform)
    

    # ===============================================================
    # Case 2: --------------Assignment 3 GAN model -> Use MNIST dataset (28×28 grayscale)------------
    # ===============================================================
    elif model_name == "GAN_Assignment3":
        # Define transforms
        transform = transforms.Compose([
                            #transforms.Resize(128),                       # Resize images to 128×128
                            #transforms.CenterCrop(64),                    # Crop the central 64×64 region
                            transforms.ToTensor(),                        # Convert PIL image → Tensor (0–1)
                            transforms.Normalize([0.5], [0.5])            # Normalize to [-1, 1] range for GANs
                        ])
        
        # Load train or test dataset depending on the flag
        dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)


    # ===============================================================
    # Case 3: ----------CNN / FCNN / EnhancedCNN / CNN_Assignment2 models -> Use CIFAR-10 dataset--------
    # ===============================================================
    elif model_name in ["FCNN", "CNN", "EnhancedCNN", "CNN_Assignment2"]:                                                        
        # Define transforms
        transform = transforms.Compose([
                        transforms.Resize((target_size, target_size)),  # Resize images if needed
                        transforms.ToTensor() # Convert image to tensor (0–1 range)
                        ])                    
     
        # Load train or test dataset depending on the flag
        dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)


    # ===============================================================
    # Case 4: ---------- Diffusion ----------
    # ===============================================================
    elif model_name == "Diffusion":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    

    # ===============================================================
    # Case 5: ---------- Energy ----------
    # ===============================================================
    elif model_name == "Energy":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)




    # ===============================================================
    # Case 6: Custom dataset (defined seperately)
    # ===============================================================
    else:
        print("NO Model defined, using CustomImageDataset class - DEVELOPMENT UNDER PROGRESS ")
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor()
        ])
        dataset = CustomImageDataset(root_dir=data_dir, transform=transform)



    # ===============================================================
    # Finally Wrap the dataset into a DataLoader
    # ===============================================================
    # DataLoader for batching for train or test data
    loader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=train,             # Shuffle only during training
                    num_workers=2,             # speeds up data loading (Optional)
                    )
    


    return loader




class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label (not used in GANs)
