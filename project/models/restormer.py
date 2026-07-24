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


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network."""

    def __init__(self, dim, ffn_expansion_factor=2.66):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.norm1 = LayerNorm2d(dim)
        self.conv_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=True)
        self.conv_in_dw = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3,
                                     padding=1, groups=hidden_dim * 2, bias=True)
        self.conv_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x1, x2 = self.conv_in_dw(self.conv_in(x)).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.conv_out(x)
        return res + x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = LinearAttention(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.ffn(x)
        return x


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                 padding=1, groups=dim * 3, bias=True)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        S = H * W
        qkv = self.qkv_dw(self.qkv(x))
        qkv = qkv.reshape(B * self.num_heads, 3, self.head_dim, S)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = q.transpose(-2, -1) * self.scale
        k = F.elu(k.transpose(-2, -1)) + 1.0
        v = F.elu(v.transpose(-2, -1)) + 1.0

        attn = torch.bmm(q, k.transpose(-2, -1))
        out = torch.bmm(attn, v)

        out = out.reshape(B, C, H, W)
        out = self.project_out(out)
        return out


class Restormer(nn.Module):
    """
    Restormer: Efficient Transformer for High-Resolution Image Restoration.
    Uses Linear Attention for O(n) complexity.
    """
    def __init__(self, in_channels=1, out_channels=1, dim=48,
                 num_blocks=None, num_refinement_blocks=4,
                 heads=None, ffn_expansion_factor=2.66):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.dim = dim

        # Level 0: Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=True)

        # Encoder
        self.level1_encoder = self._make_encoder_block(dim, heads[0], num_blocks[0], ffn_expansion_factor)
        self.downsample1 = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)

        self.level2_encoder = self._make_encoder_block(dim * 2, heads[1], num_blocks[1], ffn_expansion_factor)
        self.downsample2 = nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = self._make_encoder_block(dim * 4, heads[2], num_blocks[2], ffn_expansion_factor)

        # Decoder
        self.upsample1 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.level2_decoder = self._make_encoder_block(dim * 2, heads[1], num_blocks[3], ffn_expansion_factor)

        self.upsample2 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.level1_decoder = self._make_encoder_block(dim, heads[0], num_blocks[3], ffn_expansion_factor)

        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor)
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=True)

    def _make_encoder_block(self, dim, num_heads, num_blocks, ffn_expansion_factor):
        return nn.Sequential(*[
            TransformerBlock(dim, num_heads, ffn_expansion_factor)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        inp = x

        # Patch Embedding
        x = self.patch_embed(x)

        # Encoder
        skip1 = self.level1_encoder(x)
        x = self.downsample1(skip1)

        skip2 = self.level2_encoder(x)
        x = self.downsample2(skip2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.upsample1(x)
        x = x + skip2
        x = self.level2_decoder(x)

        x = self.upsample2(x)
        x = x + skip1
        x = self.level1_decoder(x)

        # Refinement
        x = self.refinement(x)

        # Output
        x = self.output(x) + inp

        return x
