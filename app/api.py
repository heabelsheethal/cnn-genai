import os
import random
import torch
from fastapi import FastAPI, HTTPException
from helper_lib.model import get_model
from helper_lib.data_loader import get_data_loader

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
    print("⚠️ GAN_Assignment3 generator not found (models/gan3_generator.pth missing)")

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
        ]
    }

# ----------------------------------------------------
# MODEL PREDICTION ENDPOINTS
# ----------------------------------------------------
@app.get("/predict/fcnn/random")
def predict_fcnn():
    return predict_model_random("FCNN")

@app.get("/predict/cnn/random")
def predict_cnn():
    return predict_model_random("CNN")

@app.get("/predict/enhancedcnn/random")
def predict_enhancedcnn():
    return predict_model_random("EnhancedCNN")

@app.get("/predict/cnn_assignment2/random")
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
@app.get("/generate/gan/random")
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
@app.get("/generate/gan_assignment3/random")
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