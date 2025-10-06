import torch

def evaluate_model(model, data_loader, criterion, device = 'cpu'):
    # TODO: calculate average loss and accuracy on the test dataset
    """
    Evaluate the model on test data.
    """

    model.to(device)
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            if criterion:
                total_loss += criterion(outputs, labels).item()

    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(data_loader) if criterion else None

    print(f"Test Accuracy: {accuracy:.3f}")
    if avg_loss is not None:
        print(f"Average Loss: {avg_loss:.4f}")

    return avg_loss, accuracy


