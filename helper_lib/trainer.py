# helper_lib/trainer.py

import torch
from tqdm import tqdm 
import os, pandas as pd


# ------------------------------------------------------------------
# Base supervised training
# ------------------------------------------------------------------
def train_model(model, data_loader, criterion, optimizer, device = 'cpu', epochs = 5):
    # TODO: run several iterations of the training loop (based on epochsparameter) and return the model
    """
    Train a model for a given number of epochs.

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to train.
    data_loader : torch.utils.data.DataLoader
        Provides batches of (inputs, labels) for training.
    criterion : torch.nn loss function (e.g., nn.CrossEntropyLoss)
        Used to compute how far predictions are from the true labels.
    optimizer : torch.optim optimizer (e.g., Adam, SGD)
        Updates model weights based on gradients.
    device : str, optional (default='cpu')
        Device to run training on ('cpu' or 'cuda'/'mps').
    epochs : int, optional (default=5)
        Number of times the model sees the full training dataset.
    """


    model.to(device)        # Move the model to the specified device (CPU or GPU)
    model.train()           # Set the model to training mode (enables dropout, batchnorm updates)
    datalogs = []           # Store loss & accuracy for each epoch


    # ------------------------------------------------------------------
    # Outer loop: run training for the specified number of epochs
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        running_loss = 0.0           # Track cumulative loss across batches in this epoch
        running_correct = 0          # Count of correctly predicted samples
        running_total = 0            # Total number of samples processed so far

        # tqdm shows a progress bar while looping through batches
        loader_with_progress = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}")


        # --------------------------------------------------------------
        # Inner loop: iterate over mini-batches of (inputs, labels)
        # --------------------------------------------------------------
        for batch_number, (inputs, labels) in enumerate(loader_with_progress):
            inputs = inputs.to(device)  # Move data to device
            labels = labels.to(device)  # Move data to device
            optimizer.zero_grad()                                  # Reset gradients from previous step (required each iteration)
            outputs = model(inputs)                                # Forward pass: compute model predictions
            
            # METRICS CALCULATION (Training Performance Tracking)
            _, predicted = torch.max(outputs.data, 1)               # Get predicted class index


            # FORWARD & BACKWARD PASS (Core Training Steps)
            loss = criterion(outputs, labels)                     # Compute loss between predictions and true labels
            loss.backward()                                       # Backward pass: compute gradients w.r.t. model parameters
            optimizer.step()                                      # Update model parameters using computed gradients

            
            # LOG DATA FOR TRACKING (Batch-Level Statistics)
            running_correct += (predicted == labels).sum().item()   # Count correct predictions
            running_total += labels.size(0)                         # Count total samples
            running_loss += loss.item()                             # Accumulate loss

            # Every 100 batches 
            if (batch_number % 100 == 99):
                # Update progress bar with live accuracy/loss
                loader_with_progress.set_postfix({
                    "avg acc": f"{running_correct/running_total:.3f}",
                    "avg loss": f"{running_loss/(batch_number+1):.4f}"
                })

                # Log metrics for batch
                datalogs.append({
                    "epoch": epoch + batch_number / len(data_loader),
                    "train_loss": running_loss / (batch_number + 1),
                    "train_accuracy": running_correct / running_total,
                })


        # --------------------------------------------------------------
        # End of epoch: compute & store average loss and accuracy
        # --------------------------------------------------------------
        # Log metrics for epoch
        avg_loss = running_loss / max(1, len(data_loader))
        avg_acc  = running_correct / max(1, running_total)
        datalogs.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": avg_acc
        })

        print(f"Finished epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={avg_acc:.3f}")



    os.makedirs("logs", exist_ok=True)
    pd.DataFrame(datalogs).to_csv(f"logs/{model.__class__.__name__}_train_log.csv", index=False)
    print(f"Saved training log → logs/{model.__class__.__name__}_train_log.csv")
    # ------------------------------------------------------------------
    # After all epochs
    # ------------------------------------------------------------------
    print("Finished Training")
    return model





# ------------------------------------------------------------------
# Diffusion model training
# ------------------------------------------------------------------

def train_diffusion(model, data_loader, optimizer, device='cpu', epochs=5):
    
    import torch.nn.functional as F
    model.train()
    diffusion_logs = []  # track avg_loss for each epoch
    for epoch in range(epochs):
        running_loss = 0
        for x, _ in data_loader:
            x = x.to(device)
            noise = torch.randn_like(x)
            t = torch.rand(x.size(0), device=device).view(-1,1,1,1)
            noise_rates, signal_rates = model.schedule_fn(t)
            x_noisy = signal_rates * x + noise_rates * noise
            pred_noise = model(x_noisy, t)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_ema()
            running_loss += loss.item()
        avg_loss = running_loss / max(1, len(data_loader))
        print(f"[Diffusion] Epoch {epoch+1}: loss={avg_loss:.4f}")
        diffusion_logs.append({"epoch": epoch + 1, "loss": avg_loss})

    os.makedirs("logs", exist_ok=True)
    pd.DataFrame(diffusion_logs).to_csv("logs/Diffusion_train_log.csv", index=False)
    print("Saved diffusion training log → logs/Diffusion_train_log.csv")
    return model



# ------------------------------------------------------------------
# Energy-based model training
# ------------------------------------------------------------------

def train_energy(model, data_loader, optimizer, device='cpu', epochs=10):
    import torch.nn.functional as F
    model.train()
    energy_logs = []
    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0
        for x_real, _ in data_loader:
            x_real = x_real.to(device)
            x_fake = torch.randn_like(x_real)
            e_pos, e_neg = model(x_real), model(x_fake)
            loss = e_pos.mean() - e_neg.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1
        avg_loss = running_loss / max(1, batch_count)
        energy_logs.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"[EBM] Epoch {epoch+1}: loss={avg_loss:.4f}")
    os.makedirs("logs", exist_ok=True)
    pd.DataFrame(energy_logs).to_csv("logs/Energy_train_log.csv", index=False)
    print("Saved energy training log → logs/Energy_train_log.csv")
    return model



