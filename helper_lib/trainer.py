import torch
from tqdm import tqdm 

def train_model(model, data_loader, criterion, optimizer, device = 'cpu', epochs = 10):
    # TODO: run several iterations of the training loop (based on epochsparameter) and return the model
    """
    Train a model for a given number of epochs.
    """

    model.to(device)
    model.train()
    datalogs = []

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct, running_total = 0, 0

        loader_with_progress = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_number, (inputs, labels) in enumerate(loader_with_progress):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            # Optional: update progress every 100 batches
            if (batch_number % 100 == 99):
                loader_with_progress.set_postfix({
                    "avg acc": f"{running_correct/running_total:.3f}",
                    "avg loss": f"{running_loss/(batch_number+1):.4f}"
                })

        # Log metrics for epoch
        datalogs.append({
            "epoch": epoch + 1,
            "train_loss": running_loss/len(data_loader),
            "train_accuracy": running_correct/running_total
        })

        print(f"Finished epoch {epoch+1}: loss={running_loss/len(data_loader):.4f}, "
              f"accuracy={running_correct/running_total:.3f}")

    print("Finished Training")
    return model