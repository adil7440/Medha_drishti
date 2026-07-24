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
    PyTorch Dataset for 2D MRI Slice Enhancement Training.
    Loads slice pairs (input_slice, target_slice) from stage2 preprocessed cache files (.npz).
    Resizes slices to uniform shape (target_size, target_size) for batching.
    Applies data augmentations: Rotation, Flip, Gamma, Noise, and Intensity Scaling.
    """

    def __init__(self, preprocessed_dir: str, dataset_filter: str = None, is_train: bool = True, target_size: int = 128, max_samples: int = 40, seed: int = 42):
        super().__init__()
        self.preprocessed_dir = Path(preprocessed_dir)
        self.is_train = is_train
        self.target_size = target_size
        random.seed(seed)
        np.random.seed(seed)

        all_files = sorted(list(self.preprocessed_dir.glob("*.npz")))
        if dataset_filter:
            all_files = [f for f in all_files if f.name.startswith(dataset_filter)]

        if max_samples and len(all_files) > max_samples:
            all_files = all_files[:max_samples]

        self.samples = []
        for filepath in all_files:
            try:
                data = np.load(filepath)
                orig = data["orig_slice"].astype(np.float32)
                target = data["stage_final"].astype(np.float32)

                # Scale normalize to [0, 1]
                orig_min, orig_max = np.min(orig), np.max(orig)
                if orig_max > orig_min:
                    orig = (orig - orig_min) / (orig_max - orig_min)

                target_min, target_max = np.min(target), np.max(target)
                if target_max > target_min:
                    target = (target - target_min) / (target_max - target_min)

                # Exclude blank slices (background content only)
                if np.mean(target) > 0.01:
                    # Resize to uniform dimensions (target_size, target_size)
                    orig = cv2.resize(orig, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                    target = cv2.resize(target, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                    self.samples.append((orig, target, filepath.name))
            except Exception as e:
                print(f"[Warning] Failed loading {filepath.name}: {e}")

        # Train / Val split (80% train, 20% val)
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.8)
        if self.is_train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def _augment(self, inp: np.ndarray, tgt: np.ndarray):
        """Applies consistent random spatial and intensity augmentations."""
        # 1. Random Rotation (+- 10 deg)
        if random.random() > 0.5:
            angle = random.uniform(-10.0, 10.0)
            inp = ndi.rotate(inp, angle, reshape=False, mode='nearest')
            tgt = ndi.rotate(tgt, angle, reshape=False, mode='nearest')

        # 2. Random Horizontal Flip
        if random.random() > 0.5:
            inp = np.fliplr(inp).copy()
            tgt = np.fliplr(tgt).copy()

        # 3. Random Gamma Correction
        if random.random() > 0.5:
            gamma = random.uniform(0.8, 1.2)
            inp = np.power(np.clip(inp, 0, 1), gamma)

        # 4. Random Intensity Scaling
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            inp = np.clip(inp * scale, 0, 1)

        # 5. Random Gaussian Noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.01, size=inp.shape).astype(np.float32)
            inp = np.clip(inp + noise, 0, 1)

        return inp, tgt

    def __getitem__(self, idx):
        inp_slice, tgt_slice, filename = self.samples[idx]

        if self.is_train:
            inp_slice, tgt_slice = self._augment(inp_slice, tgt_slice)

        # Convert to PyTorch Tensors (1, H, W)
        inp_tensor = torch.from_numpy(inp_slice).unsqueeze(0).float()
        tgt_tensor = torch.from_numpy(tgt_slice).unsqueeze(0).float()

        return inp_tensor, tgt_tensor, filename


def get_dataloaders(preprocessed_dir: str, batch_size: int = 4, target_size: int = 128, max_samples: int = 40, num_workers: int = 0):
    """Creates Train and Validation DataLoaders with uniform spatial resolution."""
    train_ds = MRISliceDataset(preprocessed_dir, is_train=True, target_size=target_size, max_samples=max_samples)
    val_ds = MRISliceDataset(preprocessed_dir, is_train=False, target_size=target_size, max_samples=max_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
