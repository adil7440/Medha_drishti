import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.w = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x * self.pool(x) * self.w


class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_ch = channels * dw_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1,
                               groups=dw_ch, bias=True)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_ch // 2)
        self.conv3 = nn.Conv2d(dw_ch // 2, channels, kernel_size=1, bias=True)

        ffn_ch = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_ch, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, channels, kernel_size=1, bias=True)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Spatial mixing
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        # Channel mixing
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        x = x + y * self.gamma

        return x


class NAFNet(nn.Module):
    """
    NAFNet: Nonlinear Activation Free Network for Image Restoration.
    Uses Simplified Gated Channel Attention blocks.
    """
    def __init__(self, in_channels=1, out_channels=1, width=32,
                 middle_blk_num=1, enc_blk_nums=None, dec_blk_nums=None):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]

        # Encoder
        self.head = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2

        # Bottleneck
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        # Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num in reversed(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        self.tail = nn.Conv2d(width, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        inp = x

        x = self.head(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.tail(x) + inp
        return x
