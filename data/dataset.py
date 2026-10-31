"""CIFAR-10 transforms and reproducible train/validation/test loaders."""
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image


class Cutout:
    """Mask one square patch in a tensor image."""
    def __init__(self, n_holes=1, length=16):
        self.n_holes, self.length = n_holes, length

    def __call__(self, image):
        h, w = image.shape[1], image.shape[2]
        mask = np.ones((h, w), dtype=np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
            x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        return image * image.new_tensor(mask).unsqueeze(0)


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def create_datasets(root="./data", download=True, seed=42):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        Cutout(n_holes=1, length=16),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_augmented = datasets.CIFAR10(root, train=True, download=download,
                                       transform=train_transform)
    train_clean = datasets.CIFAR10(root, train=True, download=download,
                                   transform=eval_transform)
    test_dataset = datasets.CIFAR10(root, train=False, download=download,
                                    transform=eval_transform)

    # Same deterministic split as the notebook: 45,000 train and 5,000 validation.
    # Match the notebook split: seed NumPy, then permute the 50,000
    # training examples and use the first 45,000 for training.
    np.random.seed(seed)
    indices = np.random.permutation(len(train_augmented))
    train_indices, val_indices = indices[:45000], indices[45000:]
    return (Subset(train_augmented, train_indices),
            Subset(train_clean, val_indices), test_dataset)


def create_loaders(batch_size=64, root="./data", seed=42, num_workers=2):
    train_set, val_set, test_set = create_datasets(root, seed=seed)
    common = {"batch_size": batch_size, "num_workers": num_workers,
              "pin_memory": True}
    return (
        DataLoader(train_set, shuffle=True, **common),
        DataLoader(val_set, shuffle=False, **common),
        DataLoader(test_set, shuffle=False, **common),
    )
