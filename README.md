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
├── environment.txt
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
        ├── q3_parallel_coordinates.png
        ├── q3_validation_accuracy_vs_step.png
        ├── q3_training_loss.png
        ├── q3_training_accuracy.png
        ├── q3_validation_loss.png
        └── q3_validation_accuracy.png
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

The project uses Python, PyTorch, Torchvision, NumPy, Pillow, Pandas, and Weights & Biases. Install versions that are compatible with your Python version and target machine. The exact accelerator and software environment can vary between systems.

| Component | Dependency |
|---|---|
| Python | 3.10 or later recommended |
| PyTorch | Compatible stable release |
| Torchvision | Version compatible with PyTorch |
| NumPy | Compatible stable release |
| Pillow | Compatible stable release |
| Pandas | Compatible stable release |
| Weights & Biases | Compatible stable release |

### Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch and Torchvision builds appropriate for your system. If you intend to use GPU acceleration, select a build compatible with your GPU driver and CUDA environment using the official PyTorch installation selector:
https://pytorch.org/get-started/locally/

To record the software and accelerator details of the environment in which you run the project, execute:

```bash
python environment_report.py
```

The generated report describes the environment at the time the command is run. It should not be treated as a record of a different machine or an earlier experiment session.

## Reproducing the selected Q4 configuration

From the repository root:

```bash
python main.py --activation silu --optimizer Adam --lr 0.01 --batch-size 64 --epochs 10 --seed 42
```

The CIFAR-10 dataset is downloaded to `./data` if it is not already present. The script trains the selected configuration, saves the best validation checkpoint to `checkpoints/best_vgg6_q4.pth`, and evaluates that checkpoint on the test set.

The script automatically selects an available supported device. Install a PyTorch build compatible with the hardware and software environment on which you run the experiment.

## Random seed and reproducibility

The default seed is 42. `utils/seed.py` seeds Python, NumPy, PyTorch CPU, and CUDA random number generators when CUDA is available. It also enables deterministic cuDNN behavior and disables cuDNN benchmarking.

Exact bit-for-bit reproducibility can still depend on the PyTorch version, accelerator hardware, driver/runtime versions, and nondeterministic operations. Record the environment used for each experiment with `python environment_report.py`.

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
