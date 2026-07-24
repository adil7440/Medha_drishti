import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, in_channels=1):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels

        sigma = 1.5
        gauss = torch.exp(torch.tensor(
            [-(x - window_size // 2) ** 2 / (2 * sigma ** 2) for x in range(window_size)]
        ))
        gauss = gauss / gauss.sum()
        _2d_gauss = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        kernel = _2d_gauss.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def forward(self, img1, img2):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.conv2d(img1, self.kernel, padding=self.window_size // 2, groups=self.in_channels)
        mu2 = F.conv2d(img2, self.kernel, padding=self.window_size // 2, groups=self.in_channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.kernel, padding=self.window_size // 2, groups=self.in_channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.kernel, padding=self.window_size // 2, groups=self.in_channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.kernel, padding=self.window_size // 2, groups=self.in_channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss using relu1_2, relu2_2, relu3_3, relu4_3 features.
    Converts 1-channel input to 3-channel for VGG.
    """
    def __init__(self, layer_weights=None):
        super().__init__()
        if layer_weights is None:
            layer_weights = {
                '3': 0.25,
                '8': 0.25,
                '15': 0.25,
                '22': 0.25,
            }
        self.layer_weights = layer_weights

        vgg = models.vgg16(pretrained=True)
        features = vgg.features.children()
        self.layers = nn.ModuleList()
        for i, layer in enumerate(features):
            self.layers.append(layer)
            if i == 22:
                break

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # Convert 1-channel to 3-channel
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        loss = 0.0
        pred_features = pred
        target_features = target

        for i, layer in enumerate(self.layers):
            pred_features = layer(pred_features)
            target_features = layer(target_features)
            layer_key = str(i)
            if layer_key in self.layer_weights:
                weight = self.layer_weights[layer_key]
                loss += weight * F.l1_loss(pred_features, target_features)

        return loss


class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel gradient magnitude."""
    def __init__(self):
        super().__init__()
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _compute_gradient_magnitude(self, x):
        # Pad for conv2d
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred, target):
        pred_edges = self._compute_gradient_magnitude(pred)
        target_edges = self._compute_gradient_magnitude(target)
        return F.l1_loss(pred_edges, target_edges)


class HybridLoss(nn.Module):
    """
    Hybrid Loss for MRI Enhancement:
    Total = w_char * Charbonnier + w_ssim * SSIM + w_perc * Perceptual + w_edge * Edge
    """
    def __init__(self, weights=None, eps=1e-6, layer_weights=None):
        super().__init__()
        if weights is None:
            weights = {
                "charbonnier": 0.35,
                "ssim": 0.25,
                "perceptual": 0.20,
                "edge": 0.20,
            }
        self.weights = weights
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.ssim = SSIMLoss()
        self.perceptual = PerceptualLoss(layer_weights=layer_weights)
        self.edge = EdgeLoss()

    def forward(self, pred, target):
        loss_char = self.charbonnier(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_perc = self.perceptual(pred, target)
        loss_edge = self.edge(pred, target)

        total = (
            self.weights["charbonnier"] * loss_char +
            self.weights["ssim"] * loss_ssim +
            self.weights["perceptual"] * loss_perc +
            self.weights["edge"] * loss_edge
        )
        return total
