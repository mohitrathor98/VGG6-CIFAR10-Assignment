"""One-epoch training routine."""
import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = correct = total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total
