# VGG6 on CIFAR-10

A modular PyTorch implementation of the VGG6 CIFAR-10 assignment. The project
contains the model, data pipeline, training/evaluation routines, optimizer
factory, seed helper, and a command-line training entry point.

## Architecture and data

- VGG6 configuration: `[64, 64, 'M', 128, 128, 'M']`
- 10 output classes.
- Batch normalization enabled.
- CIFAR-10 normalization:
  - mean `(0.4914, 0.4822, 0.4465)`
  - standard deviation `(0.2023, 0.1994, 0.2010)`
- Training augmentation: random crop with padding 4, random horizontal flip,
  and Cutout (one hole, length 16).
- Evaluation/test transforms: tensor conversion and normalization only.
- Deterministic split: 45,000 training images and 5,000 validation images.

## Environment

The experiments were run in Google Colab with a Tesla T4 GPU and W&B 0.28.1.
Install PyTorch and torchvision using the command recommended by the official
PyTorch selector for the target machine. Then install the remaining packages:

```bash
pip install -r requirements.txt
```

The notebook output records W&B 0.28.1. Exact versions of every other
dependency were not recorded in the supplied notebook, so this repository
does not claim unverified version pins.

## Reproduce Q4

From the repository root:

```bash
python main.py --activation silu --optimizer Adam --lr 0.01 \
  --batch-size 64 --epochs 10 --seed 42
```

CIFAR-10 is downloaded automatically into `./data` if it is not already
present. The best validation checkpoint is saved to
`checkpoints/best_vgg6_q4.pth`. The script then evaluates that checkpoint on
the test set.

## Random seed

The experiment seed is 42 by default. `utils/seed.py` seeds Python, NumPy,
PyTorch CPU and CUDA generators, and enables deterministic cuDNN behavior.
Exact bit-for-bit reproduction can still depend on library versions and
hardware.

## Assignment results

The best W&B sweep configuration was:

| Parameter | Value |
|---|---|
| Activation | SiLU |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Batch size | 64 |
| Epochs | 10 |
| Seed | 42 |
| Validation accuracy | 83.95% |

The Q4 reproduction run also reported 83.95% validation accuracy. See the
W&B sweep and run pages in the submitted report for experiment records.

## Repository layout

- `models/vgg6.py`: VGG6 architecture and activation factory.
- `data/dataset.py`: CIFAR-10 transforms, split, and data loaders.
- `training/train.py`: training loop for one epoch.
- `training/evaluate.py`: validation/test evaluation.
- `utils/optimizers.py`: six optimizer options used in the experiments.
- `utils/seed.py`: random seed setup.
- `main.py`: command-line training, best-checkpoint saving, and test evaluation.
- `checkpoints/`: model checkpoint.
- `results/`: experiment result tables.

## Notes

This code modularizes the implementation used for the assignment. The
`main.py` entry point reproduces the selected Q4 configuration; it does not
automatically run every Q2 comparison or the W&B sweep.
