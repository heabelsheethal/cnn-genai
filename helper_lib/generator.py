# helper_lib/generator.py

from .energy_model import langevin_sample
import torch
from torchvision.utils import save_image



def generate_samples_diffusion(model, device="cpu", num_samples=16, steps=50, image_size=32):
    # model is a DiffusionModel wrapping a UNet
    model.eval()
    with torch.no_grad():
        imgs = model.generate(num_images=num_samples, diffusion_steps=steps, image_size=image_size)
    return imgs


def generate_samples_energy(model, device="cpu", num_samples=16, steps=100, step_size=0.1, noise_scale=0.01):
    return langevin_sample(model, num_samples=num_samples, steps=steps,
                           step_size=step_size, noise_scale=noise_scale, device=device)



