"""VGG6 model definition used in the CIFAR-10 assignment."""
import math
import torch.nn as nn
import torch.nn.functional as F


def activation_function(name: str) -> nn.Module:
    name = name.lower()
    activations = {
        "relu": lambda: nn.ReLU(inplace=True),
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "silu": lambda: nn.SiLU(inplace=True),
        "gelu": nn.GELU,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


def make_layers(cfg, batch_norm=True, activation="relu"):
    layers, in_channels = [], 3
    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, value, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(value))
            layers.append(activation_function(activation))
            in_channels = value
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(128, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.classifier(x)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / n))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


VGG6_CONFIG = [64, 64, "M", 128, 128, "M"]


def vgg6(num_classes=10, batch_norm=True, activation="relu"):
    """Build the six-convolution-layer VGG configuration."""
    return VGG(make_layers(VGG6_CONFIG, batch_norm, activation), num_classes)
