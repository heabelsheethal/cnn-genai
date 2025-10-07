from .fcnn import MLP
from .simple_cnn import SimpleCNN
from .enhanced_cnn import EnhancedCNN
from .assignment2 import Assignment2_CNN

def get_model(model_name="FCNN"):
    if model_name == "FCNN":
        return MLP()
    elif model_name == "CNN":
        return SimpleCNN()
    elif model_name == "EnhancedCNN":
        return EnhancedCNN()
    elif model_name == "CNN_Assignment2":
        return Assignment2_CNN()
    else:
        raise ValueError(f"Unknown model: {model_name}")