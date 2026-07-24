import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for Channel Attention."""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2 + SEBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class DnCNN(nn.Module):
    """
    Ultra-Enhanced U-DnCNN: 
    U-Net style architecture with Squeeze-and-Excitation blocks 
    and residual skip connections to preserve high-frequency details.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_features: int = 64, num_layers: int = 20):
        # We ignore num_layers here as the architecture is explicitly defined.
        super(DnCNN, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, num_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(num_features, num_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(num_features * 2, num_features * 4))
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(num_features * 4, num_features * 2)
        
        self.up1 = nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(num_features * 2, num_features)
        
        # Output
        self.outc = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Decoder
        d2 = self.up2(x3)
        # Pad if necessary (though with 128/256 it's strictly powers of 2)
        diffY = x2.size()[2] - d2.size()[2]
        diffX = x2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv2(d2)
        
        d1 = self.up1(d2)
        diffY = x1.size()[2] - d1.size()[2]
        diffX = x1.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.conv1(d1)
        
        residual = self.outc(d1)
        # Residual learning
        return x - residual
