# VGG6 on CIFAR-10

**Course:** CS6886W – Systems Engineering for Deep Learning  
**Assignment:** Assignment 1  
**Institution:** Department of Computer Science and Engineering, IIT Madras

This repository contains the experiment notebook and a modular PyTorch implementation for training and evaluating a VGG6 model on CIFAR-10. It covers baseline training, activation-function comparisons, optimizer comparisons, hyperparameter experiments, W&B sweeps, and reproduction of the selected configuration.

## Repository links

- **GitHub repository:** https://github.com/mohitrathor98/VGG6-CIFAR10-Assignment
- **W&B baseline run:** https://wandb.ai/cs25m518-indian-institute-of-technology-madras/VGG6-CIFAR10-Assignment1/runs/ztcqjxc9
- **W&B sweep dashboard:** https://wandb.ai/cs25m518-indian-institute-of-technology-madras/VGG6-CIFAR10-Assignment1/sweeps/pjyaw4dz

## Project contents

```text
.
├── A1_VGG6_Experiments.ipynb
├── main.py
├── requirements.txt
├── environment_report.py
├── .gitignore
├── models/
│   └── vgg6.py
├── data/
│   └── dataset.py
├── training/
│   ├── train.py
│   └── evaluate.py
├── utils/
│   ├── optimizers.py
│   └── seed.py
├── checkpoints/
│   └── best_vgg6_q4.pth
└── results/
    ├── optimizer_results.csv
    └── plots/
        ├── q1_baseline_loss.png
        ├── q1_baseline_accuracy.png
        └── [six exported W&B charts]
```

## Model and data configuration

- VGG6 feature configuration: `[64, 64, 'M', 128, 128, 'M']`
- Number of classes: 10
- Batch normalization: enabled
- CIFAR-10 mean: `(0.4914, 0.4822, 0.4465)`
- CIFAR-10 standard deviation: `(0.2023, 0.1994, 0.2010)`
- Training augmentation: random crop (32 × 32, padding 4), random horizontal flip, tensor conversion, normalization, and Cutout (one hole, length 16).
- Validation/test preprocessing: tensor conversion and normalization, without training augmentation.
- Dataset split: 45,000 training images and 5,000 validation images from the 50,000-image CIFAR-10 training set; 10,000 test images.
- The modular data loader uses the same NumPy seed-and-permutation split procedure as the experiment notebook.

## Selected configuration and reported result

The highest validation accuracy in the W&B sweep was obtained with:

| Parameter | Selected value |
|---|---:|
| Activation | SiLU |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Batch size | 64 |
| Epochs | 10 |
| Seed | 42 |
| Best validation accuracy | 83.95% |

The Q4 reproduction reported 83.95% validation accuracy. The submitted checkpoint is `checkpoints/best_vgg6_q4.pth`.

The Q1 baseline test result was **74.98% top-1 accuracy**, with test loss **0.7438**. The baseline's best validation accuracy was 75.08% at epoch 9.

## Environment and dependencies

The experiments were run in Google Colab using a Tesla T4 GPU. W&B version `0.28.1` is recorded in the experiment environment. The exact versions of the other libraries should be captured from the original Colab runtime and recorded in `environment.txt`; do not infer versions from a different machine.

### Capture the original Colab environment

In the original Colab notebook/runtime used for the experiments, run this cell and save its output as `environment.txt` in the repository:

```python
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
```

`requirements.txt` lists the required packages. Install a PyTorch/torchvision build compatible with the target machine's CPU or CUDA environment using the official PyTorch installation selector, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

For a reproducible environment, use the recorded `environment.txt` versions when recreating the experiment environment.

## Reproducing the selected Q4 configuration

From the repository root:

```bash
python main.py --activation silu --optimizer Adam --lr 0.01 --batch-size 64 --epochs 10 --seed 42
```

The CIFAR-10 dataset is downloaded to `./data` if it is not already present. The script trains the selected configuration, saves the best validation checkpoint to `checkpoints/best_vgg6_q4.pth`, and evaluates that checkpoint on the test set.

To run on CPU, the script automatically selects CPU when CUDA is unavailable. Training time will be substantially longer than on a GPU.

## Random seed and reproducibility

The default seed is 42. `utils/seed.py` seeds Python, NumPy, PyTorch CPU, and CUDA random number generators when CUDA is available. It also enables deterministic cuDNN behavior and disables cuDNN benchmarking.

Exact bit-for-bit reproducibility can still depend on the PyTorch/CUDA versions, hardware, and nondeterministic operations. Record the runtime environment alongside results.

## Code organization

- `models/vgg6.py`: VGG6 architecture and activation-function factory.
- `data/dataset.py`: CIFAR-10 transforms, Cutout, train/validation split, and data loaders.
- `training/train.py`: one-epoch training routine.
- `training/evaluate.py`: validation/test evaluation routine.
- `utils/optimizers.py`: optimizer factory for SGD, Nesterov-SGD, Adam, Adagrad, RMSprop, and Nadam.
- `utils/seed.py`: random-seed setup.
- `main.py`: command-line training and evaluation entry point.
- `A1_VGG6_Experiments.ipynb`: experiment notebook containing the assignment experiments.
- `results/`: exported plots and optimizer results.
- `checkpoints/`: selected trained model checkpoint.

**Scope note:** `main.py` reproduces the selected Q4 configuration. The full Q2 comparisons and W&B sweep are documented in the experiment notebook and W&B dashboard; `main.py` does not launch all those experiments automatically.
