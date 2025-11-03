# helper_lib/diffusion_model.py

import math, copy, torch
import torch.nn as nn
import torch.nn.functional as F

# --- Sinusoidal Embedding ---


class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 1, 1)
        returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
        """
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)         


# --- Schedules ---

def linear_diffusion_schedule(diffusion_times, min_rate=1e-4, max_rate=0.02):
    """
    diffusion_times: Tensor of shape (T,) with values in [0, 1)
    Returns:
        noise_rates: Tensor of shape (T,)
        signal_rates: Tensor of shape (T,)
    """
    diffusion_times = diffusion_times.to(dtype=torch.float32)
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    signal_rates = torch.sqrt(alpha_bars)
    noise_rates = torch.sqrt(1.0 - alpha_bars)
    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    # diffusion_times: Tensor of shape [T] or [B] with values in [0, 1]
    signal_rates = torch.cos(diffusion_times * math.pi / 2)
    noise_rates = torch.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates

def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    # Flatten diffusion_times to handle any shape
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    # Compute start and end angles from signal rate bounds
    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))

    # Linearly interpolate angles
    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)

    # Compute signal and noise rates
    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)

    return noise_rates, signal_rates



# --- Building Blocks ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.proj(x)
        # x = self.norm(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual

class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x




# --- UNet backbone ---
class UNet(nn.Module):
    def __init__(self, image_size, num_channels, embedding_dim=32):
        super().__init__()
        self.image_size = image_size          
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim


        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(32, 32, kernel_size=1)  # embedding_dim = 32
            

        # ---- Correct order: DownBlock(width, block_depth, in_channels) ----
        # --- Encoder (downsampling) ---
        # After concatenation, input becomes 32 (image) + 32 (embedding) = 64 channels
        self.down1 = DownBlock(64, 2, 64)  # width=64, block_depth=2, in_channels=64
        self.down2 = DownBlock(128, 2, 64) # width=128, block_depth=2, in_channels=64
        self.down3 = DownBlock(256, 2, 128) # width=256, block_depth=2, in_channels=128

        # --- Bottleneck ---
        self.mid1 = ResidualBlock(256, 256)
        self.mid2 = ResidualBlock(256, 256)

        # ---- Correct order: UpBlock(width, block_depth, in_channels) ----
        # --- Decoder (upsampling) ---
        # Decoder (upsampling) — width = skip channels, in_channels = decoder channels entering the block
        self.up1 = UpBlock(width=256, block_depth=2, in_channels=256)  # (256 upsampled + 256 skip) -> 256
        self.up2 = UpBlock(width=128, block_depth=2, in_channels=256)  # (256 upsampled + 128 skip) -> 128
        self.up3 = UpBlock(width=64,  block_depth=2, in_channels=128)  # (128 upsampled + 64  skip) -> 64

        self.final = nn.Conv2d(64, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)  # shape: (B, 1, 1, 32)
        # Upsample to match image size like TF reference
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips) 
        x = self.down3(x, skips)    

        x = self.mid1(x)     
        x = self.mid2(x)   

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)


# --- Diffusion wrapper (keep only architecture + inference logic) ---

class DiffusionModel(nn.Module):
    def __init__(self, model, schedule_fn):
        super().__init__()
        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_network.eval()
        self.ema_decay = 0.8
        self.schedule_fn = schedule_fn
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def to(self, device):
        # Override to() to ensure both networks move to the same device
        super().to(device)
        self.ema_network.to(device)
        return self

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # Use EMA network for inference, main network for training
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()

        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images


    def forward(self, x, t):
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        noise_rates, signal_rates = self.schedule_fn(t)
        pred_noises, _ = self.denoise(x, noise_rates, signal_rates, training=True)
        return pred_noises


    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)

            # Debug generation process
            if step % max(1, diffusion_steps // 4) == 0:  # Print 4 times during generation
                print(f"Generation Step {step}/{diffusion_steps}: t={1-step*step_size:.3f}")
                print(f"  Current images std: {current_images.std().item():.4f}")
                print(f"  Pred images std: {pred_images.std().item():.4f}")
                print(f"  Signal rate: {signal_rates.mean().item():.4f}, Noise rate: {noise_rates.mean().item():.4f}")

            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, image_size=64, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn((num_images, self.network.num_channels, image_size, image_size), device=next(self.parameters()).device)
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

 
    def update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1.0 - self.ema_decay) * param)



