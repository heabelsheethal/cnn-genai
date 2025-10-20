import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt


# ==============================================================
# Generator (for MNIST 28×28 grayscale)
# ==============================================================
class Generator(nn.Module):
    def __init__(self, z_dim=100):                            
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.Linear(z_dim, 7 * 7 * 128),                                   # Fully connected layer -> 7×7×128       
            nn.Unflatten(1, (128, 7, 7)),                                    # Reshape to (128, 7, 7)       
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # ConvTranspose2D: 128->64, kernel size 4, stride 2, padding 1, output 14×14
            nn.BatchNorm2d(64),                                              # BatchNorm 
            nn.ReLU(True),                                                   # ReLU 
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),   # ConvTranspose2D: 64->1, output 28×28
            nn.Tanh()                                                        # Tanh activation for Output in [-1,1]
        )

    def forward(self, z):
        return self.model(z)


# ==============================================================
# Discriminator (for MNIST 1×28×28)
# ==============================================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),   # Conv2D: 1->64, kernel size 4, stride 2, padding 1, output 14×14
            nn.LeakyReLU(0.2, inplace=True),                        # LeakyReLU(0.2) activation
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Conv2D: 64->128, kernel size 4, stride 2, padding 1, output 7×7
            nn.BatchNorm2d(128),                                    # BatchNorm
            nn.LeakyReLU(0.2, inplace=True),                        # LeakyReLU(0.2) activation
            nn.Flatten(),                                           # Flatten to vector
            nn.Linear(128 * 7 * 7, 1),                              # Linear layer to get a single output
            nn.Sigmoid()                                            # Output = probability of real/fake
        )

    def forward(self, x):
        return self.model(x)


# ==============================================================
# Training Loop (Standard GAN, not WGAN)
# ==============================================================
def train_gan_model(gen, disc, data_loader, device='cpu', z_dim=100,
                    lr=0.0002, beta1=0.5, epochs=5, show_images=True):
    criterion = nn.BCELoss()
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))

    gen.to(device)
    disc.to(device)

    fixed_noise = torch.randn(64, z_dim, device=device)
    datalogs = []

    for epoch in range(epochs):
        loader_with_progress = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_number, (real, _) in enumerate(loader_with_progress):
            real = real.to(device)
            batch_size = real.size(0)
            noise = torch.randn(batch_size, z_dim, device=device)          # z_dim = 100 generates random noise (batch_size, 100)

            # -------------------------------
            # Train Discriminator
            # -------------------------------
            fake = gen(noise)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            disc_real = disc(real).view(-1, 1)
            disc_fake = disc(fake.detach()).view(-1, 1)

            loss_real = criterion(disc_real, real_labels)
            loss_fake = criterion(disc_fake, fake_labels)
            loss_disc = (loss_real + loss_fake) / 2

            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # -------------------------------
            # Train Generator
            # -------------------------------
            disc_fake = disc(fake).view(-1, 1)
            loss_gen = criterion(disc_fake, real_labels)

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_number % 100 == 0:
                loader_with_progress.set_postfix({
                    "D loss": f"{loss_disc.item():.4f}",
                    "G loss": f"{loss_gen.item():.4f}"
                })
                datalogs.append({
                    "epoch": epoch + batch_number / len(data_loader),
                    "D loss": loss_disc.item(),
                    "G loss": loss_gen.item(),
                })

        # -------------------------------
        # Visualization & Save Generated Images
        # -------------------------------
        if show_images:
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
            grid = make_grid(fake, normalize=True)

            # --- Display (optional) ---
            plt.figure(figsize=(6, 6))
            plt.imshow(grid.permute(1, 2, 0), cmap='gray')
            plt.title(f"Epoch {epoch+1}")
            plt.axis("off")
            plt.show()

            # --- Save ---
            save_dir = os.path.join("img", "generated_image_assignment3_gan")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}.png")
            plt.imsave(save_path, grid.permute(1, 2, 0).numpy(), cmap='gray')
            print(f"Saved generated image: {save_path}")

    print(" GAN (Assignment 3) Training Completed Successfully")
    return gen, disc, datalogs


# ==============================================================
# Wrapper Class for get_model()
# ==============================================================
class GANModel:
    def __init__(self, z_dim=100):
        self.z_dim = z_dim
        self.gen = Generator(z_dim)
        self.critic = Discriminator()

    def get_models(self):
        return self.gen, self.critic
    
