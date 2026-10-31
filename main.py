"""Train the selected VGG6 configuration and save its best validation checkpoint."""
import argparse
import os
import torch
import torch.nn as nn

from data.dataset import create_loaders
from models.vgg6 import vgg6
from training.train import train_one_epoch
from training.evaluate import evaluate
from utils.seed import set_seed
from utils.optimizers import get_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", default="silu")
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--checkpoint", default="./checkpoints/best_vgg6_q4.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = create_loaders(
        args.batch_size, root=args.data_root, seed=args.seed
    )

    model = vgg6(activation=args.activation).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, args.lr)
    best_val_accuracy = -1.0

    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "best_val_accuracy": best_val_accuracy,
                "epoch": epoch + 1,
            }, args.checkpoint)

    # Evaluate the selected checkpoint on the untouched test set.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
