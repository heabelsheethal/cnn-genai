import torch
from tqdm import tqdm 

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
        datalogs.append({
            "epoch": epoch + 1,
            "train_loss": running_loss/len(data_loader),
            "train_accuracy": running_correct/running_total
        })

        print(f"Finished epoch {epoch+1}: loss={running_loss/len(data_loader):.4f}, "
              f"accuracy={running_correct/running_total:.3f}")



    # ------------------------------------------------------------------
    # After all epochs
    # ------------------------------------------------------------------
    print("Finished Training")
    return model