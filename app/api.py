# app/api.py

import os
import random
import torch
from fastapi import FastAPI, HTTPException
from helper_lib.model import get_model
from helper_lib.data_loader import get_data_loader
from helper_lib.generator import generate_samples_diffusion, generate_samples_energy
from torchvision.utils import save_image

# ----------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------
app = FastAPI(title="CNN GenAI API")

# ----------------------------------------------------
# DEVICE SETUP
# ----------------------------------------------------
device = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print(f"Using device: {device}")

# ====================================================
# 1️⃣ LOAD CLASSIFICATION MODELS (CIFAR-10 based)
# ====================================================
model_names = ["FCNN", "CNN", "EnhancedCNN", "CNN_Assignment2"]
models = {}

for name in model_names:
    model = get_model(name).to(device)
    model_path = f"models/{name}.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[name] = model
        print(f"Loaded {name} model successfully")
    except FileNotFoundError:
        print(f"⚠️ Warning: {model_path} not found. Using untrained {name} model")

# ----------------------------------------------------
# LOAD TEST IMAGES (PER MODEL SIZE)
# ----------------------------------------------------
test_images = {}
for name in model_names:
    if name.lower() == "fcnn":
        loader = get_data_loader("data/test", batch_size=1, train=False, model_name="FCNN")
    elif name.lower() == "cnn_assignment2":
        loader = get_data_loader("data/test", batch_size=1, train=False, model_name="CNN_Assignment2")
    else:
        loader = get_data_loader("data/test", batch_size=1, train=False, model_name="CNN")
    test_images[name] = list(loader)

print("Test images preloaded successfully.")

# ====================================================
# 2️⃣ PRELOAD GAN MODELS
# ====================================================

gan_models = {}

# ---- WGAN (CelebA) ----
try:
    from helper_lib.gan import Generator as WGANGenerator
    wgen = WGANGenerator(100).to(device)
    wgen.load_state_dict(torch.load("models/gan_generator.pth", map_location=device))
    wgen.eval()
    gan_models["GAN"] = wgen
    print("Loaded GAN (CelebA) generator successfully")
except FileNotFoundError:
    print("GAN generator not found (models/gan_generator.pth missing)")

# ---- Assignment 3 GAN (MNIST) ----
try:
    from helper_lib.assignment3_gan import Generator as MNISTGenerator
    gen3 = MNISTGenerator(100).to(device)
    gen3.load_state_dict(torch.load("models/gan3_generator.pth", map_location=device))
    gen3.eval()
    gan_models["GAN_Assignment3"] = gen3
    print("Loaded GAN_Assignment3 (MNIST) generator successfully")
except FileNotFoundError:
    print("GAN_Assignment3 generator not found (models/gan3_generator.pth missing)")

# ====================================================
# 3️⃣ PRELOAD DIFFUSION & ENERGY MODELS
# ====================================================
diffusion_model = None
energy_model = None

# ---- Diffusion (CIFAR-10 32×32 RGB) ----
try:
    diffusion_model = get_model("Diffusion").to(device)
    diffusion_model.load_state_dict(torch.load("models/diffusion_model.pth", map_location=device))
    diffusion_model.eval()
    print("Loaded Diffusion model successfully")
except FileNotFoundError:
    print("Diffusion model weights not found (models/diffusion_model.pth missing) — API will use untrained weights.")
    try:
        diffusion_model = get_model("Diffusion").to(device)
        diffusion_model.eval()
    except Exception as e:
        print(f"Failed to init Diffusion model: {e}")

# ---- Energy-based model (CIFAR-10 32×32 RGB) ----
try:
    energy_model = get_model("Energy").to(device)
    energy_model.load_state_dict(torch.load("models/energy_model.pth", map_location=device))
    energy_model.eval()
    print("Loaded Energy model successfully")
except FileNotFoundError:
    print("Energy model weights not found (models/energy_model.pth missing) — API will use untrained weights.")
    try:
        energy_model = get_model("Energy").to(device)
        energy_model.eval()
    except Exception as e:
        print(f"Failed to init Energy model: {e}")

# ----------------------------------------------------
# ROOT ENDPOINT
# ----------------------------------------------------
@app.get("/")
def read_root():
    return {
        "message": "Welcome to CNN GenAI API!",
        "available_routes": [
            "/predict/fcnn/random",
            "/predict/cnn/random",
            "/predict/enhancedcnn/random",
            "/predict/cnn_assignment2/random",
            "/generate/gan/random",
            "/generate/gan_assignment3/random",
            "/generate/diffusion/random",
            "/generate/energy/random",
        ]
    }

# ----------------------------------------------------
# MODEL PREDICTION ENDPOINTS
# ----------------------------------------------------
@app.get("/predict/fcnn/random", tags=["CNN"])
def predict_fcnn():
    return predict_model_random("FCNN")

@app.get("/predict/cnn/random", tags=["CNN"])
def predict_cnn():
    return predict_model_random("CNN")

@app.get("/predict/enhancedcnn/random", tags=["CNN"])
def predict_enhancedcnn():
    return predict_model_random("EnhancedCNN")

@app.get("/predict/cnn_assignment2/random", tags=["CNN"])
def predict_assignment2():
    return predict_model_random("CNN_Assignment2")

# ----------------------------------------------------
# HELPER FUNCTION (for classification models)
# ----------------------------------------------------
def predict_model_random(model_name: str):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    image, label = random.choice(test_images[model_name])
    image = image.to(device)
    label = label.to(device)

    with torch.no_grad():
        outputs = models[model_name](image)
        _, predicted = torch.max(outputs, 1)

    return {
        "model": model_name,
        "predicted_class": int(predicted.item()),
        "true_class": int(label.item())
    }

# ==============================================================
# GAN IMAGE GENERATION ENDPOINTS
# ==============================================================

# ---- WGAN (CelebA) ----
@app.get("/generate/gan/random", tags=["GAN"])
def generate_wgan_image():
    import torchvision.utils as vutils
    if "GAN" not in gan_models:
        raise HTTPException(status_code=404, detail="GAN generator not preloaded")

    gen = gan_models["GAN"]
    z_dim = 100

    noise = torch.randn(1, z_dim, 1, 1, device=device)
    with torch.no_grad():
        fake_img = gen(noise).cpu()

    os.makedirs("generated_images", exist_ok=True)
    save_path = "generated_images/sample_gan.png"
    vutils.save_image(fake_img, save_path, normalize=True)
    print(f"Saved new GAN image → {save_path}")

    return {"message": "New CelebA GAN image generated!", "file_path": save_path}

# ---- Assignment 3 GAN (MNIST) ----
@app.get("/generate/gan_assignment3/random", tags=["GAN"])
def generate_assignment3_gan_image():
    import torchvision.utils as vutils
    if "GAN_Assignment3" not in gan_models:
        raise HTTPException(status_code=404, detail="Assignment3 GAN generator not preloaded")

    gen = gan_models["GAN_Assignment3"]
    z_dim = 100

    noise = torch.randn(1, z_dim, device=device)
    with torch.no_grad():
        fake_img = gen(noise).cpu()

    os.makedirs("generated_images", exist_ok=True)
    save_path = "generated_images/sample_assignment3_gan.png"
    vutils.save_image(fake_img, save_path, normalize=True)
    print(f"Saved new Assignment3 GAN image → {save_path}")

    return {"message": "New MNIST GAN image generated!", "file_path": save_path}

# ---- Diffusion (CIFAR-10) ----
@app.get("/generate/diffusion/random", tags=["Diffusion & Energy"])
def generate_diffusion_image():
    if diffusion_model is None:
        raise HTTPException(status_code=404, detail="Diffusion model not preloaded")
    os.makedirs("generated_images", exist_ok=True)
    try:
        imgs = generate_samples_diffusion(
            diffusion_model, device=str(device), num_samples=4, steps=20, image_size=32
        )
        save_path = "generated_images/sample_diffusion.png"
        save_image(imgs, save_path, nrow=4, normalize=True)
        print(f"Saved new Diffusion image grid → {save_path}")
        return {"message": "New Diffusion images generated!", "file_path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diffusion generation failed: {e}")

# ---- Energy-based model (CIFAR-10) ----
@app.get("/generate/energy/random", tags=["Diffusion & Energy"])
def generate_energy_image():
    if energy_model is None:
        raise HTTPException(status_code=404, detail="Energy model not preloaded")
    os.makedirs("generated_images", exist_ok=True)
    try:
        imgs = generate_samples_energy(
            energy_model, device=str(device), num_samples=4, steps=50
        )
        save_path = "generated_images/sample_energy.png"
        save_image(imgs, save_path, nrow=4, normalize=True)
        print(f"Saved new Energy image grid → {save_path}")
        return {"message": "New Energy images generated!", "file_path": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Energy generation failed: {e}")
    

# ---------------------------------------------
# LLM ROUTES
# ---------------------------------------------
from app.api_llm import router as llm_router
app.include_router(llm_router)