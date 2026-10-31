"""Evaluation routine for validation and test datasets."""
import torch


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = correct = total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, 100.0 * correct / total
