"""Print package and accelerator versions for reproducibility records."""
import sys
import torch
import torchvision
import numpy
import PIL
import pandas
import wandb

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("NumPy:", numpy.__version__)
print("Pillow:", PIL.__version__)
print("Pandas:", pandas.__version__)
print("Weights & Biases:", wandb.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
