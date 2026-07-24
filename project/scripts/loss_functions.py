import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    PyTorch Differentiable Structural Similarity Index (SSIM) Loss.
    """
    def __init__(self, window_size: int = 11, in_channels: int = 1):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels

        # Create 1D Gaussian kernel
        sigma = 1.5
        gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / (2 * sigma ** 2) for x in range(window_size)]))
        gauss = gauss / gauss.sum()

        # Create 2D Gaussian kernel
        _2d_gauss = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        kernel = _2d_gauss.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        mu1 = F.conv2d(img1, self.kernel, padding=self.window_size // 2, groups=self.in_channels)
        mu2 = F.conv2d(img2, self.kernel, padding=self.window_size // 2, groups=self.in_channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.kernel, padding=self.window_size // 2, groups=self.in_channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.kernel, padding=self.window_size // 2, groups=self.in_channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.kernel, padding=self.window_size // 2, groups=self.in_channels) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class HybridLoss(nn.Module):
    """
    Hybrid Loss Function for MRI Image Enhancement:
    Loss = l1_weight * L1 + ssim_weight * SSIM_Loss
    Default: 0.8 * L1 + 0.2 * SSIM_Loss
    """
    def __init__(self, l1_weight: float = 0.8, ssim_weight: float = 0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_val = self.l1(pred, target)
        ssim_val = self.ssim(pred, target)
        return self.l1_weight * l1_val + self.ssim_weight * ssim_val
