# helper_lib/evaluator.py

import torch

def evaluate_model(model, data_loader, criterion, device = 'cpu'):
    # TODO: calculate average loss and accuracy on the test dataset
    """
    Evaluate the model on test data.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained neural network model to be evaluated.
    data_loader : torch.utils.data.DataLoader
        DataLoader for the test/validation dataset.
    criterion : torch.nn loss function (e.g., nn.CrossEntropyLoss)
        Used to compute the average test loss (optional).
    device : str, optional (default='cpu')
        Device on which to perform evaluation ('cpu' or 'cuda'/'mps').

    Returns:
    --------
    avg_loss : float or None
        Average test loss (if criterion is provided).
    accuracy : float
        Model accuracy on the test dataset.
    """

    model.to(device)   # Move model to device (CPU/GPU)
    model.eval()       # Set eval mode (disable dropout, batchnorm)
    total_correct = 0  # Total correct predictions
    total_samples = 0  # Total test samples
    total_loss = 0.0   # Total accumulated loss

    # Disable gradient computation during evaluation (saves memory & time)
    with torch.no_grad():

        for images, labels in data_loader:    # Loop through batches of (images, labels) from the test DataLoader
            images = images.to(device)       # Move data to device
            labels = labels.to(device)       # Move labels to device
            outputs = model(images)                             # Forward pass: compute model outputs/predictions
            _, predicted = torch.max(outputs.data, 1)           # Get predicted class index
            total_correct += (predicted == labels).sum().item() # Count how many predictions match the true labels
            total_samples += labels.size(0)                     # Update total sample count

            # If a loss function is provided, accumulate batch loss
            if criterion:
                total_loss += criterion(outputs, labels).item()

    accuracy = total_correct / total_samples                          # overall accuracy
    avg_loss = total_loss / len(data_loader) if criterion else None   # average loss across all batches (if criterion provided)

    # Print evaluation results
    print(f"Test Accuracy: {accuracy:.3f}")    
    if avg_loss is not None:
        print(f"Average Loss: {avg_loss:.4f}")

    return avg_loss, accuracy


