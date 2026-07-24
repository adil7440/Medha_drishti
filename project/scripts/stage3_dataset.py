import os
import random
import cv2
import numpy as np
import scipy.ndimage as ndi
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class MRISliceDataset(Dataset):
    """
    Enhanced PyTorch Dataset for 2D MRI Slice Enhancement Training.
    Loads slice pairs from stage2 preprocessed cache files (.npz).
    Supports full augmentation pipeline including elastic deformation and affine transforms.
    """

    def __init__(self, preprocessed_dir: str, is_train: bool = True,
                 target_size: int = 128, max_samples: int = None,
                 seed: int = 42, aug_config: dict = None):
        super().__init__()
        self.preprocessed_dir = Path(preprocessed_dir)
        self.is_train = is_train
        self.target_size = target_size
        self.aug_config = aug_config or {}
        random.seed(seed)
        np.random.seed(seed)

        all_files = sorted(list(self.preprocessed_dir.glob("*.npz")))

        if max_samples and len(all_files) > max_samples:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(all_files), size=max_samples, replace=False)
            all_files = [all_files[i] for i in sorted(indices)]

        self.samples = []
        for filepath in all_files:
            try:
                data = np.load(filepath)
                orig = data["orig_slice"].astype(np.float32)
                target = data["stage_final"].astype(np.float32)

                orig_min, orig_max = np.min(orig), np.max(orig)
                if orig_max > orig_min:
                    orig = (orig - orig_min) / (orig_max - orig_min)

                target_min, target_max = np.min(target), np.max(target)
                if target_max > target_min:
                    target = (target - target_min) / (target_max - target_min)

                if np.mean(target) > 0.01:
                    orig = cv2.resize(orig, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                    target = cv2.resize(target, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                    self.samples.append((orig, target, filepath.name))
            except Exception:
                continue

        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.8)
        if self.is_train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def _augment(self, inp: np.ndarray, tgt: np.ndarray):
        """Applies consistent random augmentations to both input and target."""
        aug = self.aug_config

        # Random Rotation
        if aug.get("random_rotation", {}).get("enabled", True):
            max_angle = aug.get("random_rotation", {}).get("max_angle", 15.0)
            if random.random() > 0.5:
                angle = random.uniform(-max_angle, max_angle)
                inp = ndi.rotate(inp, angle, reshape=False, mode='nearest')
                tgt = ndi.rotate(tgt, angle, reshape=False, mode='nearest')

        # Random Affine
        if aug.get("random_affine", {}).get("enabled", True):
            if random.random() > 0.5:
                degrees = aug.get("random_affine", {}).get("degrees", 10)
                translate = aug.get("random_affine", {}).get("translate", [0.05, 0.05])
                scale_range = aug.get("random_affine", {}).get("scale", [0.95, 1.05])
                angle = random.uniform(-degrees, degrees)
                tx = random.uniform(-translate[0], translate[0]) * inp.shape[1]
                ty = random.uniform(-translate[1], translate[1]) * inp.shape[0]
                scale = random.uniform(scale_range[0], scale_range[1])
                M = cv2.getRotationMatrix2D((inp.shape[1] / 2, inp.shape[0] / 2), angle, scale)
                M[0, 2] += tx
                M[1, 2] += ty
                inp = cv2.warpAffine(inp, M, (inp.shape[1], inp.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                tgt = cv2.warpAffine(tgt, M, (tgt.shape[1], tgt.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Elastic Deformation
        if aug.get("elastic_deformation", {}).get("enabled", True):
            if random.random() > 0.5:
                alpha = aug.get("elastic_deformation", {}).get("alpha", 200.0)
                sigma = aug.get("elastic_deformation", {}).get("sigma", 10.0)
                shape = inp.shape
                dx = ndi.gaussian_filter(np.random.randn(*shape) * alpha, sigma)
                dy = ndi.gaussian_filter(np.random.randn(*shape) * alpha, sigma)
                y_grid, x_grid = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
                indices_y = np.clip(y_grid + dy, 0, shape[0] - 1).astype(np.int32)
                indices_x = np.clip(x_grid + dx, 0, shape[1] - 1).astype(np.int32)
                inp = inp[indices_y, indices_x]
                tgt = tgt[indices_y, indices_x]

        # Random Horizontal Flip
        if random.random() > 0.5:
            inp = np.fliplr(inp).copy()
            tgt = np.fliplr(tgt).copy()

        # Random Vertical Flip
        if random.random() > 0.5:
            inp = np.flipud(inp).copy()
            tgt = np.flipud(tgt).copy()

        # Random Gamma
        if aug.get("random_gamma", {}).get("enabled", True):
            gamma_range = aug.get("random_gamma", {}).get("gamma_range", [0.7, 1.5])
            if random.random() > 0.5:
                gamma = random.uniform(gamma_range[0], gamma_range[1])
                inp = np.power(np.clip(inp, 0, 1), gamma)

        # Random Intensity Scaling
        if aug.get("random_intensity_scaling", {}).get("enabled", True):
            scale_range = aug.get("random_intensity_scaling", {}).get("scale_range", [0.8, 1.2])
            if random.random() > 0.5:
                scale = random.uniform(scale_range[0], scale_range[1])
                inp = np.clip(inp * scale, 0, 1)

        # Random Contrast
        if aug.get("random_contrast", {}).get("enabled", True):
            factor_range = aug.get("random_contrast", {}).get("factor_range", [0.7, 1.3])
            if random.random() > 0.5:
                factor = random.uniform(factor_range[0], factor_range[1])
                mean = np.mean(inp)
                inp = np.clip((inp - mean) * factor + mean, 0, 1)

        # Random Brightness
        if aug.get("random_brightness", {}).get("enabled", True):
            brightness_range = aug.get("random_brightness", {}).get("brightness_range", [-0.15, 0.15])
            if random.random() > 0.5:
                brightness = random.uniform(brightness_range[0], brightness_range[1])
                inp = np.clip(inp + brightness, 0, 1)

        # Random Gaussian Noise
        if aug.get("random_gaussian_noise", {}).get("enabled", True):
            std_range = aug.get("random_gaussian_noise", {}).get("std_range", [0.005, 0.02])
            if random.random() > 0.5:
                std = random.uniform(std_range[0], std_range[1])
                noise = np.random.normal(0, std, size=inp.shape).astype(np.float32)
                inp = np.clip(inp + noise, 0, 1)

        # Random Cropping
        if aug.get("random_cropping", {}).get("enabled", True) and self.is_train:
            crop_ratio = aug.get("random_cropping", {}).get("crop_ratio", 0.9)
            if random.random() > 0.5:
                h, w = inp.shape
                crop_h = int(h * crop_ratio)
                crop_w = int(w * crop_ratio)
                y = random.randint(0, h - crop_h)
                x = random.randint(0, w - crop_w)
                inp = inp[y:y + crop_h, x:x + crop_w]
                tgt = tgt[y:y + crop_h, x:x + crop_w]
                inp = cv2.resize(inp, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
                tgt = cv2.resize(tgt, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        return inp.astype(np.float32), tgt.astype(np.float32)

    def __getitem__(self, idx):
        inp_slice, tgt_slice, filename = self.samples[idx]

        if self.is_train:
            inp_slice, tgt_slice = self._augment(inp_slice, tgt_slice)

        inp_tensor = torch.from_numpy(inp_slice).unsqueeze(0).float()
        tgt_tensor = torch.from_numpy(tgt_slice).unsqueeze(0).float()

        return inp_tensor, tgt_tensor, filename


def get_dataloaders(preprocessed_dir: str, batch_size: int = 4,
                    target_size: int = 128, max_samples: int = None,
                    num_workers: int = 0, aug_config: dict = None):
    """Creates Train and Validation DataLoaders."""
    train_ds = MRISliceDataset(preprocessed_dir, is_train=True, target_size=target_size,
                                max_samples=max_samples, aug_config=aug_config)
    val_ds = MRISliceDataset(preprocessed_dir, is_train=False, target_size=target_size,
                              max_samples=max_samples, aug_config=aug_config)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
