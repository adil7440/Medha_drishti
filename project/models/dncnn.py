import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN: Deep Convolutional Neural Network for MRI Artifact and Noise Removal.
    Residual learning framework predicting additive residual noise map.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_features: int = 64, num_layers: int = 17):
        super(DnCNN, self).__init__()

        layers = []
        # Input layer: Conv + ReLU
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Intermediate layers: Conv + BatchNorm + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # Output layer: Conv
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual learning formulation: Enhanced = Input - Residual
        residual = self.dncnn(x)
        return x - residual
