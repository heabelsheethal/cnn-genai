import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model

def main():
    # -------------------------
    # Command line argument
    # -------------------------
    parser = argparse.ArgumentParser(description="Select model to train")
    parser.add_argument("--model", type=str, default="CNN", 
                        choices=["FCNN", "CNN", "EnhancedCNN"],
                        help="Choose which model to train")
    args = parser.parse_args()

    # -------------------------
    # Device setup
    # -------------------------
    device = torch.device('mps') if torch.backends.mps.is_available() else \
             torch.device('cuda') if torch.cuda.is_available() else \
             torch.device('cpu')
    print(f"Using device: {device}")

    # -------------------------
    # Load data
    # -------------------------
    train_loader = get_data_loader('data/train', batch_size=64)
    test_loader = get_data_loader('data/test', batch_size=64, train=False)

    # -------------------------
    # Select model
    # -------------------------
    model = get_model(args.model)

    # -------------------------
    # Loss function & optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -------------------------
    # Train model
    # -------------------------
    trained_model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=5)

    # -------------------------
    # Evaluate model
    # -------------------------
    evaluate_model(trained_model, test_loader, criterion, device=device)

if __name__ == "__main__":
    main()