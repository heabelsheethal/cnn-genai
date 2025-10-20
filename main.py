import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.gan import train_gan_model as train_gan_model_v1
from helper_lib.assignment3_gan import train_gan_model as train_gan_model_v2


# Centralize the available model names
MODEL_CHOICES = ["FCNN", "CNN", "EnhancedCNN", "CNN_Assignment2", "GAN", "GAN_Assignment3"]

def main():
    # -------------------------Command line argument-------------------------
    parser = argparse.ArgumentParser(description="Select model to train")
    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_CHOICES,
        help="Choose which model to train"
    )
    args = parser.parse_args()

    # If no model is given, ask interactively
    if args.model is None:
        print("\nAvailable models:")
        for i, name in enumerate(MODEL_CHOICES, 1):
            print(f"{i}. {name}")

        # Keep asking until a valid numeric choice is made
        while True:
            user_input = input("\nEnter the number of the model you want to train: ").strip()
            if not user_input.isdigit():
                print("WRONG CHOICE. Choose from available models.")
                continue

            choice = int(user_input)
            if 1 <= choice <= len(MODEL_CHOICES):
                args.model = MODEL_CHOICES[choice - 1]
                break
            else:
                print("WRONG CHOICE. Choose from available models.")

    
    # -------------------------Device setup-------------------------
    device = torch.device('mps') if torch.backends.mps.is_available() else \
             torch.device('cuda') if torch.cuda.is_available() else \
             torch.device('cpu')
    print(f"Using device: {device}")

    # -------------------------Load data-------------------------
    train_loader = get_data_loader('data/train', model_name=args.model)
    test_loader = get_data_loader('data/test', train=False, model_name=args.model)




    # -------------------------Model setup & Training-------------------------
    if args.model == "GAN":
        # ---------------- GAN Models ----------------
        model_gen, model_crit = get_model(args.model)
        
        # ------------------------- Train model using custom GAN loop-------------------------
        trained_gen, trained_crit, logs = train_gan_model_v1(
                                                        model_gen, 
                                                        model_crit, 
                                                        train_loader, 
                                                        device=device, 
                                                        epochs=1,
                                                        n_critic=1,      # fewer critic steps
                                                        show_images=False,
                                                        fast_mode=True   # enables quick debug training
                                                        )
        
        # ------------------------- Save trained GAN models -------------------------
        # Save both networks separately
        os.makedirs("models", exist_ok=True)
        torch.save(trained_gen.state_dict(), "models/gan_generator.pth")
        torch.save(trained_crit.state_dict(), "models/gan_critic.pth")
        print("Saved GAN Generator → models/gan_generator.pth")
        print("Saved GAN Critic → models/gan_critic.pth")

        # ------------------------- Evaluate model -------------------------
        # GANs don't use a classification evaluator, so skip evaluate_model()
        print("Skipping evaluation (GAN has no accuracy metric).")

    elif args.model == "GAN_Assignment3":
        model_gen, model_crit = get_model(args.model)

        trained_gen, trained_crit, logs = train_gan_model_v2(
                                                    model_gen,
                                                    model_crit,
                                                    train_loader,
                                                    device=device,
                                                    epochs=5,
                                                    show_images=True
                                                )

        os.makedirs("models", exist_ok=True)
        torch.save(trained_gen.state_dict(), "models/gan3_generator.pth")
        torch.save(trained_crit.state_dict(), "models/gan3_critic.pth")
        print("Saved GAN3 Generator → models/gan3_generator.pth")
        print("Saved GAN3 Critic → models/gan3_critic.pth")
        print("Skipping evaluation (GAN has no accuracy metric).")

    else: 
        # ---------------- CNN based Models ----------------
        model = get_model(args.model)
        criterion = nn.CrossEntropyLoss()                      # Loss function for classification
        optimizer = optim.Adam(model.parameters(), lr=0.001)   # optimizer
        # ------------------------- Train model-------------------------
        trained_model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=1)

        # ------------------------- Save trained model -------------------------
        os.makedirs("models", exist_ok=True)  # ensure folder exists
        model_path = f"models/{args.model}.pth"
        torch.save(trained_model.state_dict(), model_path)
        print(f"Saved trained model to {model_path}")

        # -------------------------Evaluate model-------------------------
        evaluate_model(trained_model, test_loader, criterion, device=device)


if __name__ == "__main__":
    main()


