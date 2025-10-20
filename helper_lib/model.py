from .fcnn import MLP
from .simple_cnn import SimpleCNN
from .enhanced_cnn import EnhancedCNN
from .assignment2 import Assignment2_CNN
from .gan import GANModel
from .assignment3_gan import GANModel as Assignment3_GANModel


def get_model(model_name="FCNN"):
    if model_name == "FCNN":
        return MLP()
    elif model_name == "CNN":
        return SimpleCNN()
    elif model_name == "EnhancedCNN":
        return EnhancedCNN()
    elif model_name == "CNN_Assignment2":
        return Assignment2_CNN()
    elif model_name == "GAN":                    
        gan_model = GANModel()
        return gan_model.gen, gan_model.critic
    elif model_name == "GAN_Assignment3":                    
        gan = Assignment3_GANModel(z_dim=100)
        return gan.gen, gan.critic                    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    



