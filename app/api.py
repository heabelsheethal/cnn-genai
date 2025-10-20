import random
from fastapi import FastAPI, HTTPException
import torch
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

# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------
model_names = ["FCNN", "CNN", "EnhancedCNN", "CNN_Assignment2"]
models = {}

for name in model_names:
    model = get_model(name).to(device)
    try:
        model.load_state_dict(torch.load(f"models/{name}.pth", map_location=device))
        model.eval()
        models[name] = model
        print(f"Loaded {name} model successfully")
    except FileNotFoundError:
        print(f"Warning: {name}.pth not found. Using untrained model")

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
        # CNN and EnhancedCNN likely expect 32x32
        loader = get_data_loader("data/test", batch_size=1, train=False, model_name="CNN")
    test_images[name] = list(loader)

# ----------------------------------------------------
# ROOT ENDPOINT
# ----------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to CNN GenAI API!"}

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

    # Each item from test_images is a batch of size 1
    image, label = random.choice(test_images[model_name])

    # image shape is [1, C, H, W] 
    # Just send to device 
    image = image.to(device)
    label = label.to(device)

    with torch.no_grad():
        outputs = models[model_name](image)  # shape [1, num_classes]
        _, predicted = torch.max(outputs, 1)  # shape [1]

    return {
        "model": model_name,
        "predicted_class": int(predicted.item()),
        "true_class": int(label.item())
    }



# ==============================================================
# 🧠 GAN IMAGE GENERATION ENDPOINTS
# ==============================================================

# ----------------------------------------------------
# WGAN (CelebA) - from helper_lib.gan
# ----------------------------------------------------
@app.get("/generate/gan/random")
def generate_wgan_image():
    from helper_lib.gan import Generator
    import torchvision.utils as vutils
    import os

    z_dim = 100
    gen = Generator(z_dim).to(device)
    model_path = "models/gan_generator.pth"

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Trained WGAN generator not found")

    # Load generator
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()

    # Generate new fake image
    noise = torch.randn(1, z_dim, 1, 1, device=device)
    with torch.no_grad():
        fake_img = gen(noise).cpu()

    os.makedirs("generated_images", exist_ok=True)
    save_path = "generated_images/sample_gan.png"
    vutils.save_image(fake_img, save_path, normalize=True)

    return {"message": "New CelebA GAN image generated!", "file_path": save_path}


# ----------------------------------------------------
# Assignment 3 GAN (MNIST) - from helper_lib.assignment3_gan
# ----------------------------------------------------
@app.get("/generate/gan_assignment3/random")
def generate_assignment3_gan_image():
    from helper_lib.assignment3_gan import Generator
    import torchvision.utils as vutils
    import os

    z_dim = 100
    gen = Generator(z_dim).to(device)
    model_path = "models/gan3_generator.pth"

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Trained Assignment3 GAN generator not found")

    # Load generator
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()

    # Generate MNIST-like fake image
    noise = torch.randn(1, z_dim, device=device)
    with torch.no_grad():
        fake_img = gen(noise).cpu()

    os.makedirs("generated_images", exist_ok=True)
    save_path = "generated_images/sample_assignment3_gan.png"
    vutils.save_image(fake_img, save_path, normalize=True)

    return {"message": "New MNIST GAN image generated!", "file_path": save_path}

