import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction, num_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(num_channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.cbam = CBAM(channels)

    def forward(self, x):
        res = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.cbam(out)
        return res + out


class MultiScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.branch2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False)
        self.branch3 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, bias=False)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = self.fuse(torch.cat([b1, b2, b3], dim=1))
        return x + self.relu(out)


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class MIRNetv2(nn.Module):
    """
    MIRNet-v2: Multi-scale Residual Network for Image Restoration.
    Features multi-scale residual blocks with channel and spatial attention.
    """
    def __init__(self, in_channels=1, out_channels=1, features=None, num_blocks_per_group=2):
        super().__init__()
        if features is None:
            features = [32, 64, 128]

        f1, f2, f3 = features[0], features[1], features[2]

        # Encoder
        self.head = nn.Conv2d(in_channels, f1, kernel_size=3, padding=1, bias=False)

        self.group1 = self._make_group(f1, num_blocks_per_group)
        self.group2 = self._make_group(f2, num_blocks_per_group)
        self.group3 = self._make_group(f3, num_blocks_per_group)

        self.down1 = Downsample(f1, f2)
        self.down2 = Downsample(f2, f3)

        self.multi_scale = MultiScaleBlock(f3)

        self.up2 = Upsample(f3, f2)
        self.up1 = Upsample(f2, f1)

        self.group2_decode = self._make_group(f2, num_blocks_per_group)
        self.group1_decode = self._make_group(f1, num_blocks_per_group)

        self.skip1 = nn.Conv2d(f1, f1, kernel_size=1, bias=False)
        self.skip2 = nn.Conv2d(f2, f2, kernel_size=1, bias=False)
        self.skip3 = nn.Conv2d(f3, f3, kernel_size=1, bias=False)

        self.tail = nn.Conv2d(f1, out_channels, kernel_size=3, padding=1, bias=False)

    def _make_group(self, channels, num_blocks):
        return nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        inp = x

        x = self.head(x)
        s1 = self.skip1(x)
        x = self.group1(x)

        x = self.down1(x)
        s2 = self.skip2(x)
        x = self.group2(x)

        x = self.down2(x)
        s3 = self.skip3(x)
        x = self.group3(x)

        x = self.multi_scale(x)

        x = self.up2(x) + s2
        x = self.group2_decode(x)

        x = self.up1(x) + s1
        x = self.group1_decode(x)

        return self.tail(x) + inp
