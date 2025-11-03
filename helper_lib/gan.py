# helper_lib/gan.py
 
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Critic (Discriminator)
# --------------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            # Input: (3, 64, 64)
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.reshape = lambda x: x.view(x.size(0), z_dim, 1, 1)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.reshape(x)
        return self.model(x)


# --------------------------------------------------------------------
# WGAN Training Loop
# --------------------------------------------------------------------
def train_gan_model(gen, critic, data_loader, device='cpu',
                    z_dim=100, lr=5e-5, n_critic=5, clip_value=0.01, epochs=20, show_images=True, fast_mode=False):
    """
    Train a Wasserstein GAN (WGAN) on CelebA dataset.
    Based on Professor's official implementation.
    """
    
    # ==========================================================
    # FAST MODE for testing
    # ==========================================================
    if fast_mode:
        print("Fast mode for testing")
        epochs = min(epochs, 2)
        n_critic = 1
        show_images = False
        torch.manual_seed(42)



    torch.manual_seed(42)
    gen.to(device)
    critic.to(device)
    gen.train()
    critic.train()

    opt_gen = optim.RMSprop(gen.parameters(), lr=lr)
    opt_critic = optim.RMSprop(critic.parameters(), lr=lr)
    fixed_noise = torch.randn(64, z_dim, 1, 1).to(device)

    datalogs = []

    for epoch in range(epochs):
        loader_with_progress = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_number, (real, _) in enumerate(loader_with_progress):
            real = real.to(device)
            batch_size = real.size(0)

            # === Train Critic ===
            for _ in range(n_critic):
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake = gen(noise).detach()
                loss_critic = -(critic(real).mean() - critic(fake).mean())

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                # Weight clipping
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # === Train Generator ===
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)
            loss_gen = -critic(fake).mean()

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_number % 100 == 0:
                loader_with_progress.set_postfix({
                    "D loss": f"{loss_critic.item():.4f}",
                    "G loss": f"{loss_gen.item():.4f}",
                })
                datalogs.append({
                    "epoch": epoch + batch_number / len(data_loader),
                    "D loss": loss_critic.item(),
                    "G loss": loss_gen.item(),
                })

        # === Visualization & Save Generated Images ===
        if show_images:
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
            grid = make_grid(fake, normalize=True)

            # --- Display (optional) ---
            plt.figure(figsize=(6, 6))
            plt.imshow(grid.permute(1, 2, 0))
            plt.title(f"Epoch {epoch+1}")
            plt.axis("off")
            plt.show()

            # --- Save ---
            save_dir = os.path.join("img", "generated_image_gan")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}.png")
            plt.imsave(save_path, grid.permute(1, 2, 0).numpy())
            print(f"Saved generated image: {save_path}")

            print("GAN Training Completed Successfully")
            return gen, critic, datalogs


# --------------------------------------------------------------------
# GAN Model Wrapper (For model.py)
# --------------------------------------------------------------------
class GANModel:
    """Wrapper to initialize both Generator and Critic for integration with get_model()."""
    def __init__(self, z_dim=100):
        self.z_dim = z_dim
        self.gen = Generator(z_dim)
        self.critic = Critic()

    def get_models(self):
        return self.gen, self.critic
    


    