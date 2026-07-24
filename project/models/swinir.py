import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """Multi-Layer Perceptron used in Swin Transformer Block."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """Partitions feature map into local non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reconstructs feature map from local windows."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention (W-MSA) with relative position bias."""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block (STB)."""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 2)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x.permute(0, 2, 3, 1)

        x = self.norm1(shortcut)
        
        # Pad to multiple of window_size if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        Hp, Wp = H + pad_h, W + pad_w

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # Reverse window
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x.permute(0, 3, 1, 2)


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)."""
    def __init__(self, dim, num_heads, window_size=8, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        for block in self.blocks:
            x = block(x)
        x = self.conv(x)
        return res + x


class SwinIRSmall(nn.Module):
    """
    SwinIR Small: Lightweight Swin Transformer for Medical Image Restoration.
    Optimized for NVIDIA RTX 3050 Laptop GPU (4GB VRAM).
    """
    def __init__(self, in_channels=1, out_channels=1, embed_dim=48, num_heads=4, window_size=8):
        super().__init__()

        # 1. Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # 2. Deep feature extraction with 3 RSTB blocks
        self.rstb1 = RSTB(dim=embed_dim, num_heads=num_heads, window_size=window_size, depth=2)
        self.rstb2 = RSTB(dim=embed_dim, num_heads=num_heads, window_size=window_size, depth=2)
        self.rstb3 = RSTB(dim=embed_dim, num_heads=num_heads, window_size=window_size, depth=2)

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # 3. High quality reconstruction
        self.conv_last = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Initial Shallow Features
        x_first = self.conv_first(x)
        
        # Deep Transformer Features
        res = self.rstb1(x_first)
        res = self.rstb2(res)
        res = self.rstb3(res)
        res = self.conv_after_body(res)
        
        # Global Residual Connection
        res = res + x_first
        
        # Reconstruction Output
        out = self.conv_last(res)
        return x + out
