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
# HELPER FUNCTION
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